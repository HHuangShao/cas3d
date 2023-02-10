# -*- coding: utf-8 -*-
import torch
import time
import numpy as np
import random

def generate_cascades(config):
    
    #this function refer to 'CasFlow: Exploring Hierarchical Structures and Uncertainty for Cascade Prediction
    #https://github.com/Xovee/casflow.
    
    # observation and prediction time settings:
    # for twitter dataset, we use 3600*24*1 (86400, 1 day) or 3600*24*2 (172800, 2 days) as observation time
    #                      we use 3600*24*32 (2764800, 32 days) as prediction time
    # for weibo   dataset, we use 1800 (0.5 hour) or 3600 (1 hour) as observation time
    #                      we use 3600*24 (86400, 1 day) as prediction time
    # a list to save the cascades
    cascades_lines = list()
    cascades_type = dict()  # 0 for train, 1 for val, 2 for test
    cascades_time_dict = dict()
    cascades_total = 0
    cascades_valid_total = 0

    # Weibo dataset: 18 for t_o of 0.5 hour and 19 for t_o of 1 hour
    if config.observation_time == 3600:
        end_hour = 19
    else:
        end_hour = 18
        
    dataset_file_path = 'data/'+config.dataset+'/dataset.txt'
    
    # Max_depth = 0
    with open(dataset_file_path) as file:
        for line in file:
            # split the cascades into 5 parts
            # 1: cascade id
            # 2: user/item id
            # 3: publish date/time
            # 4: number of adoptions
            # 5: a list of adoptions
            cascades_total += 1
            parts = line.split('\t')
            cascade_id = parts[0]
            
            
            # filter cascades by their publish date/time
            if config.dataset == 'weibo':
                # timezone invariant
                hour = int(time.strftime('%H', time.gmtime(float(parts[2])))) + 8
                if hour < 8 or hour >= end_hour:
                    continue
            elif config.dataset == 'twitter':
                month = int(time.strftime('%m', time.localtime(float(parts[2]))))
                day = int(time.strftime('%d', time.localtime(float(parts[2]))))
                if month == 4 and day > 10:
                    continue
            else:
                pass

            paths = parts[4].strip().split(' ')

            observation_path = list()
            # number of observed popularity
            p_o = 0
            for p in paths:
                # observed adoption/participant
                nodes = p.split(':')[0].split('/')
                
                # if len(nodes)>Max_depth:
                #     Max_depth=len(nodes)
                
                time_now = int(p.split(':')[1])
                if time_now < config.observation_time:
                    p_o += 1
                # save observed adoption/participant into 'observation_path'
                observation_path.append((nodes, time_now))

            # filter cascades which observed popularity less than 10
            if p_o < 10:
                continue

            # sort list by their publish time/date
            observation_path.sort(key=lambda tup: tup[1])

            # for each cascade, save its publish time into a dict
            cascades_time_dict[cascade_id] = int(parts[2])

            o_path = list()

            for i in range(len(observation_path)):
                nodes = observation_path[i][0]
                t = observation_path[i][1]
                o_path.append('/'.join(nodes) + ':' + str(t))

            # write data into the targeted file, if they are not excluded
            line = parts[0] + '\t' + parts[1] + '\t' + parts[2] + '\t' \
                   + parts[3] + '\t' + ' '.join(o_path) + '\n'
            cascades_lines.append(line)
            cascades_valid_total += 1
        
    shuffle_time = list(cascades_time_dict.keys())
    random.seed(0)
    random.shuffle(shuffle_time)
    
    # open three files to save train, val, and test set, respectively
    count = 0
    
    for key in shuffle_time:
        if count < cascades_valid_total * .7:
            cascades_type[key] = 0  # training set, 70%
        elif count < cascades_valid_total * .85:
            cascades_type[key] = 1  # validation set, 15%
        else:
            cascades_type[key] = 2  # test set, 15%
        count += 1
    
    data_train_size = 0
    data_val_size = 0
    data_test_size = 0
            
    train_X = []
    train_Y = []
    val_X = []
    val_Y = []
    test_X = []
    test_Y = []
    
    for line in cascades_lines:
        # split the message into 5 parts as we just did
        parts = line.split('\t')
        
        cascade_id = parts[0]
        
        #DDD starts at 2 
        x = [[0 for _ in range(config.Maxdepth-2)] for _ in range(config.time_interval_num)]
        
        nodes_time = dict()
        
        nodes_time['-1']=0
        
        p_y = 0
        o_y = 0
        
        paths = parts[4].split(' ')
        for p in paths:
            nodes = p.split(':')[0].split('/')
            time_now = int(p.split(":")[1])
            
            nodes_time[nodes[-1]] = time_now
            
            if len(nodes)==1:
                p_y+=1
                o_y+=1
                continue
            if time_now < config.pred_time:
                p_y+=1
                if time_now < config.observation_time :
                    o_y+=1
                    
                    #If the diffusion depth is greater than the maximum
                    if len(nodes)>config.Maxdepth:
                        depth_index = config.Maxdepth-2
                    else:
                        #DDD starts at 2, and the index of the diffusion depth equal to 2 is 0
                        depth_index = len(nodes)-2
                    
                    time_now_index = int(time_now//config.time_interval)
                    
                    x[time_now_index][depth_index]+=1
                
                    
        y=p_y-o_y
        
        ##Note that in the test set the target output is converted back to log2(y),instead of log2(y+1)  !!!
        
        # 0 to train, 1 to validate, 2 to test    
        if cascade_id in cascades_type and cascades_type[cascade_id] == 0:
            data_train_size+=1
            train_X.append(x)
            train_Y.append(np.log2(y+1))
        elif cascade_id in cascades_type and cascades_type[cascade_id] == 1:
            data_val_size+=1
            val_X.append(x)
            val_Y.append(np.log2(y+1))
        elif cascade_id in cascades_type and cascades_type[cascade_id] == 2:
            data_test_size+=1
            test_X.append(x)
            test_Y.append(np.log2(y+1))
                
    if torch.cuda.is_available():
        train_X = torch.tensor(train_X,dtype=torch.float).cuda()
        train_Y = torch.tensor(train_Y,dtype=torch.float).view(-1,1).cuda()
        val_X = torch.tensor(val_X,dtype=torch.float).cuda()
        val_Y = torch.tensor(val_Y,dtype=torch.float).view(-1,1).cuda()
        test_X = torch.tensor(test_X,dtype=torch.float).cuda()
        test_Y = torch.tensor(test_Y,dtype=torch.float).view(-1,1).cuda()
    else:
        train_X = torch.tensor(train_X,dtype=torch.float)
        train_Y = torch.tensor(train_Y,dtype=torch.float).view(-1,1)
        val_X = torch.tensor(val_X,dtype=torch.float)
        val_Y = torch.tensor(val_Y,dtype=torch.float).view(-1,1)
        test_X = torch.tensor(test_X,dtype=torch.float)
        test_Y = torch.tensor(test_Y,dtype=torch.float).view(-1,1)
        
    print("Total:{} train:{} val:{} test:{}   ".format(cascades_valid_total,data_train_size,data_val_size,data_test_size))
    return [train_X,val_X,test_X],[train_Y,val_Y,test_Y]

def save_result(filename,res):
    
    with open(filename,'a') as fw:
        
        fw.write(res)
        
    return