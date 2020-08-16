# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt


class Mask(torch.autograd.Function):
    
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return torch.tensor([input[0], input[1]])
    
    @staticmethod
    def backward(self, grad_output):
        
        input, = self.saved_tensors
        
        grad_input = grad_output.clone()
        
        willReturn = torch.tensor([[grad_input[0], grad_input[1], 0, 0]]).reshape((4,1))
        
        return willReturn
        
        
        


x = torch.rand(4,1, requires_grad = True)

b = x.clone()

print(x)

learning_rate = 1e-4

losses = []

mask = Mask.apply

for i in range(int(1e4)):
    
    a = torch.pow(mask(x), 2)
    
    b = torch.sqrt(torch.sum(a))
    
    loss = torch.pow(2-b , 2)
    
    losses.append(loss.item())  
    
    
    
    loss.backward()
    
    with torch.no_grad():
        
        x -= x.grad * learning_rate
        
        x.grad.zero_()


plt.plot(losses)