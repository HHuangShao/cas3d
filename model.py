import torch
from loss import Loss

class Cas3D(torch.nn.Module):
    def __init__(self,config):
        
        super(Cas3D, self).__init__()
        
        self.config = config
        
        self.lr = config.lr
        
        self.batch_size = config.batch_size
        
        self.epochs = config.epochs
        
        self.input_size = config.Maxdepth-2
        
        self.hidden_size = config.hidden_size
        
        self.num_layers = config.num_layers
        
        self.mlp = [self.hidden_size]+config.mlp
        
        # self.mlp = [self.input_size*self.config.time_interval_num]+config.mlp
        
        ###############################################################
        
        self.gru = torch.nn.GRU(self.input_size,self.hidden_size,bidirectional=False,batch_first=True,num_layers=self.num_layers)
        
        # self.lstm = torch.nn.LSTM(self.input_size,self.hidden_size,bidirectional=False,batch_first=True,num_layers=self.num_layers)
        
        mlp_layers = []
        
        for i in range(1,len(self.mlp)): 
                mlp_layers+=[
                    torch.nn.Linear(self.mlp[i-1], self.mlp[i]),
                    torch.nn.ReLU(inplace=True)
                    ]
        self.mlp = torch.nn.Sequential(*mlp_layers[:-1])
            
        self.opt = torch.optim.Adam(list(self.parameters()), lr=self.lr)
        
        if torch.cuda.is_available():
            
            self = self.cuda()
        
    def forward(self,x):
        outs,hn = self.gru(x)
        # outs,hn = self.lstm(x)
        x = hn[-1]
        # x = hn[0][-1]
        y = self.mlp(x)
        # y = self.mlp(x.flatten(start_dim=1))
        return y
    
    def train_model(self,X,Y):
        
        loss_mse = Loss('mse')
        
        loss_msle = Loss('msle')
        
        loss_mape = Loss('mape')
        
        dataset_train = torch.utils.data.TensorDataset(X[0],Y[0])
        
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = self.batch_size,num_workers=0,shuffle=True)
        
        dataset_val = torch.utils.data.TensorDataset(X[1],Y[1])
        
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = self.batch_size,num_workers=0,shuffle=False)
        
        dataset_test = torch.utils.data.TensorDataset(X[2],Y[2])
        
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = self.batch_size,num_workers=0,shuffle=False)
        
        k=0
        best_result = [0]*6
        
        tolerance = 0 
        
        while(k < self.epochs and tolerance < 10):  
            self.train()
            batch = 0
            train_total_msle = 0
            # train_total_mape = 0
            for x,y in dataloader_train:
                self.train()
                self.opt.zero_grad()
                train_val = self(x)
                msle_loss = loss_mse(train_val,y)
                # mape_loss = loss_mape(train_val,y)
                loss = msle_loss
                loss.backward()
                train_total_msle+=msle_loss.item()
                # train_total_mape+=mape_loss.item()
                self.opt.step()
                batch+=1
                print('\r epochs:{}/{} batch:{}/{}  train:{:.4f} val:{:.4f} test:{:.4f}/{:.4f}                           '.format(k+1,self.epochs,batch, len(dataloader_train),
                        best_result[0],
                        # best_result[1],
                        best_result[2],
                        # best_result[3],
                        best_result[4],
                        best_result[5],
                        ),end='')
            self.eval()
            train_msle = train_total_msle/len(dataloader_train)
            # train_mape = train_total_mape/len(dataloader_train)
            
            val_total_msle = 0
            # val_total_mape = 0
            for x,y in dataloader_val:
                val_val = self(x)
                val_total_msle+=loss_mse(val_val,y).item()
                # val_total_mape += loss_mape(val_val,y).item()
            val_msle = val_total_msle/len(dataloader_val)
            # val_mape = val_total_mape/len(dataloader_val)
            
            test_total_msle = 0
            test_total_mape = 0
            for x,y in dataloader_test:
                test_val = self(x)
                
                # Extract y from log(y+1) like CasFlow
                test_val[test_val<0]=0
                test_val = torch.pow(2,test_val).sub(1)
                test_val[test_val<1]=1
                y=torch.pow(2,y).sub(1)
                y[y<1]=1
                
                # print(test_val,y)
                test_total_msle += loss_msle(test_val,y).item()
                test_total_mape += loss_mape(test_val,y).item()
            test_msle = test_total_msle/len(dataloader_test)
            test_mape = test_total_mape/len(dataloader_test)
            
            if best_result[0] ==0 or best_result[2] > val_msle:
                tolerance = 0
                best_result[0] = train_msle
                # best_result[1] = train_mape
                best_result[2] = val_msle
                # best_result[3] = val_mape
                best_result[4] = test_msle
                best_result[5] = test_mape
                torch.save(self.state_dict(), 'model_state_dict.pth')
            else:
                tolerance += 1
            print('\r epochs:{}/{} batch:{}/{}  train:{:.4f} val:{:.4f} test:{:.4f}/{:.4f}          '.format(k+1,self.epochs,batch, len(dataloader_train),
                        best_result[0],
                        # best_result[1],
                        best_result[2],
                        # best_result[3],
                        best_result[4],
                        best_result[5],
                        ),end='')
            k+=1
        print('\n')
        state_dict = torch.load('model_state_dict.pth')
        self.load_state_dict(state_dict)     
        self.eval()
                    
        return best_result[0],best_result[2],best_result[4],best_result[5]
