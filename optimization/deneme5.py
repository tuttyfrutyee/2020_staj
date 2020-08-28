# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

dtype_torch = torch.float64


def generateRandomCovariance_positiveDefinite(dim):
    randomCovariance = np.random.randn(dim, dim)
    randomCovariance = 0.5*(randomCovariance + randomCovariance.T) / ( np.max(abs(randomCovariance))) + dim * np.eye(dim)
    return randomCovariance

def generateRandomCovariances_positiveDefinite(n, dim):
    covariances = []
    for i in range(n):
        covariances.append(generateRandomCovariance_positiveDefinite(dim))
    return np.array(covariances, dtype="float")



zs = torch.rand(10,2,1)

Q = torch.from_numpy(generateRandomCovariance_positiveDefinite(5)).requires_grad_(True)
MeasurementNoise = torch.from_numpy(generateRandomCovariance_positiveDefinite(2))

xs = []

x0 = torch.rand(5,1)
P0 = torch.eye(5,5)


x_ = x.clone()





def f_predict_model1(x, dt):
    
    x_new = x.clone()
    x_new2 = x.clone()
    
    x_neww = x_new2[3] * dt * torch.cos(x_new2[2])
    y_neww = x_new2[3] * dt * torch.sin(x_new2[2])
    
    x_new[0] = x_neww
    x_new[1] = y_neww
    
    return x_new

def h_measure_model1(x):
    x_ = x.clone()
    return x_[0:2]


def putAngleInRange(angle):
    
    angle = angle % (2*np.pi)
    
    if(angle > (np.pi)):
        angle -= 2*np.pi
    elif(angle < (-np.pi)):
        angle += 2*np.pi
        
    return angle

def normalizeState(x):
    
    normalizedPhi = putAngleInRange(x[2].data)
    normalizedPhiDot = putAngleInRange(x[4].data)
    
    x[2] = normalizedPhi
    x[4] = normalizedPhiDot
    
    

def h_measure_model1(x):
    return x[0:2]


print(x)

learning_rate = 1e-4

dt = 0.1

losses = []

xs = []

    F = torch.array([      [1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]       ], dtype = dtype_torch)   
    
for i in range(int(5)):
    
    x_predict = f_predict_model1(x, dt)
    P_predict = 
    
    print(x)
    
    y_predict = f_predict_model1(x_, dt)
    
    loss = torch.sum(torch.pow(y - y_predict , 2))
    
    losses.append(loss.item())  
    
    
    
    loss.backward()
    
    with torch.no_grad():
        
        x -= x.grad * learning_rate
        
        x.grad.zero_()
        
    x_ = x.clone()
    
    normalizeState(x_)


plt.plot(losses)
