# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

dtype_torch = torch.float64


x = torch.rand(5,1, requires_grad = True)
y = torch.rand(5,1)


# class f_predict_model1(torch.autograd.Function):
    
#     @staticmethod
#     def forward(self, input):
#         self.save_for_backward(input)
        
#         x0 = torch.tensor(x[0], requires_grad = True, dtype=dtype_torch)
#         x1 = torch.tensor(x[1], requires_grad = True, dtype=dtype_torch)
        
        
#         x = input.clone()
        
#         x_new = x[0] + x[3] * dt * torch.cos(x[2])
#         y_new = x[1] + x[3] * dt * torch.sin(x[2])
        
#         x[0] = x_new
#         x[1] = y_new        
        
#         return x
    
#     @staticmethod
#     def backward(self, grad_output):
        
#         input, = self.saved_tensors
        
#         grad_input = grad_output.clone()
        
#         willReturn = torch.tensor([[grad_input[0], grad_input[1], 0, 0]]).reshape((4,1))
        
#         return willReturn
        

def f_predict_model1(x, dt):
    
    x_new = x.clone()
    x_new2 = x.clone()
    
    x_neww = x_new2[0] + x_new2[3] * dt * torch.cos(x_new2[2])
    y_neww = x_new2[1] + x_new2[3] * dt * torch.sin(x_new2[2])
    
    x_new[0] = x_neww
    x_new[1] = y_neww
    
    return x_new



def h_measure_model1(x):
    return x[0:2]


print(x)

learning_rate = 1e-4
dt = 0.1

losses = []


for i in range(int(1e4)):
    
    y_predict = f_predict_model1(x, dt)
    
    loss = torch.sum(torch.pow(y - y_predict , 2))
    
    losses.append(loss.item())  
    
    
    
    loss.backward()
    
    with torch.no_grad():
        
        x -= x.grad * learning_rate
        
        x.grad.zero_()


plt.plot(losses)
