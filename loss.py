# -*- coding: utf-8 -*-
import torch

class Loss(torch.nn.Module):
    
    def __init__(self,loss_type):
        
        super(Loss, self).__init__()
        
        self.loss_type = loss_type

    def forward(self, Input, target):
        
        if self.loss_type =='mse':
            
            return (Input-target).pow(2).mean()
        
        elif self.loss_type =='msle':
            
            Input = Input.log2()
            
            target = target.log2()
            
            return (Input-target).pow(2).mean()
        
        elif self.loss_type =='mape':
            
            # Input = Input.add(1).log2()
            
            # target = target.add(1).log2()
            
            return (Input.add(1).log2()-target.add(1).log2()).abs().div(target.add(2).log2()).mean()
        
        return None
