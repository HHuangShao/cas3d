import math
class Config():
    def __init__(self,dataset,pred_time,observation_time,time_interval,hidden_size,num_layers,mlp,lr,batch_size,epochs=1000):
        
        self.dataset = dataset
        
        self.observation_time = observation_time
        
        self.time_interval = time_interval
        
        self.pred_time = pred_time
        
        self.mlp = mlp
        
        self.batch_size = batch_size
        
        self.lr = lr
        
        if dataset=='twitter':
            self.Maxdepth = 5
        if dataset=='weibo':
            self.Maxdepth = 30
           
        self.time_interval_num = math.ceil(observation_time/time_interval)
        
        self.hidden_size = hidden_size
        
        self.num_layers = num_layers
        
        self.epochs = epochs
        
    def toString(self):
        s = "pt:{} ot:{} tn:{} mlp:{} hs:{} nl:{} b:{} lr:{}".format(self.pred_time,self.observation_time,self.time_interval_num,
        '-'.join([str(m) for m in self.mlp]),
        self.hidden_size,
        self.num_layers,
        self.batch_size,
        self.lr)
        return s