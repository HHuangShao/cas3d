# -*- coding: utf-8 -*-
from utils import generate_cascades,save_result
from model import Cas3D
from config import Config
from itertools import product
import os
import sys
import time
import torch

if sys.argv.__len__() > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
elif torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
dataset='weibo'

def main(dataset):
    
    if len(sys.argv)>2 and sys.argv[1]=='weibo':
        ##Weibo    
        Hpyer_parameters = dict(
            dataset = ['weibo'],
            K = [10],
            predict_time = [24*3600],
            Observation_time = [1*3600],
            time_interval = [36],
            hidden_size = [64],
            num_layers = [1,2,3],
            mlp = [[64,32,1]],
            lr = [0.005],
            batch_size = [64],
            )
    elif len(sys.argv)>2:
        ##      Twitter   
        Hpyer_parameters = dict(
            dataset = ['twitter'],
            K = [10],
            predict_time = [3600*24*32],
            Observation_time = [1*24*3600],
            time_interval = [900],
            hidden_size = [100],
            num_layers = [1,2,3],
            mlp = [[64,32,1]],
            lr = [0.005],
            batch_size = [64],
            )
    elif  dataset=='weibo':
        Hpyer_parameters = dict(
            datasets = [dataset],
            K = [10],
            predict_time = [3600*24],
            Observation_time = [0.5*3600],
            time_interval = [120],
            hidden_size = [100],
            num_layers = [2],
            mlp = [[64,32,1]],
            lr = [0.005],
            batch_size = [64],
            )
    elif  dataset=='twitter':
        Hpyer_parameters = dict(
            dataset = ['twitter'],
            K = [10],
            predict_time = [3600*24*32],
            Observation_time = [1*24*3600],
            time_interval = [1800],
            hidden_size = [100],
            num_layers = [2],
            mlp = [[64,32,1]],
            lr = [0.005],
            batch_size = [64],
            )
    
    for dataset,K,pred_time,Observation_time,time_interval_num,hidden_size,num_layer,mlp,lr,batch_size in product(*list(Hpyer_parameters.values())):
        
        config = Config(dataset,pred_time,Observation_time,time_interval_num,hidden_size,num_layer,mlp,lr,batch_size)
        
        propress_start = time.time()
        
        X,Y = generate_cascades(config)
        
        propress_time = int(time.time()-propress_start)
        
        for _ in range(K):
            
            train_start = time.time()
            
            model = Cas3D(config)
                
            train_msle,val_msle,test_msle,test_mape = model.train_model(X,Y)
            
            train_time = int(time.time()-train_start)
            
            save_result(dataset+"_result.txt","{:.3f}/{:.3f}/{:.3f}-{:.3f} P/T:{}/{} ".format(train_msle,val_msle,test_msle,test_mape,propress_time,train_time)+config.toString()+"\n")
        
        save_result(dataset+"_result.txt",'\n')

if __name__ == '__main__':
    
    main(dataset)






