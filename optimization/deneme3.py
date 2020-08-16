# -*- coding: utf-8 -*-
import torch
import numpy as np


dtype_torch = torch.float64
dtype_numpy = np.float

processNoise = torch.eye(4, dtype=dtype_torch)


def generateRandomCovariance_positiveDefinite(dim):
    randomCovariance = np.random.randn(dim, dim)
    randomCovariance = 0.5*(randomCovariance + randomCovariance.T) / ( np.max(abs(randomCovariance))) + dim * np.eye(dim)
    
    return torch.from_numpy(randomCovariance)


def f_predict_model0(x, P, dt):

    F = np.array([      [1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]       ], dtype = dtype_numpy)   

    F = torch.from_numpy(F)            

    x_predict = torch.mm(F, x)
    P_predict = torch.mm(F, torch.mm(P, F.T)) + processNoise
    
    
    return (x_predict, P_predict)


P = generateRandomCovariance_positiveDefinite(4)
P.requires_grad = True
x = torch.rand((4,1), dtype = dtype_torch, requires_grad = True)
dt = 0.1

x_should = torch.rand((4,1), dtype = dtype_torch)
P_should = torch.rand((4,4), dtype = dtype_torch)

learning_rate = 1e-3

for i in range(1000):

    x_predict, P_predict = f_predict_model0(x, P, dt)
    
    P_predict = torch.cholesky(P_predict)
    
    loss1 = torch.sum(torch.pow(x_should - x_predict, 2))
    loss2 = torch.sum(torch.pow(P_should - P_predict, 2))
    
    loss1.backward()
    loss2.backward()

    with torch.no_grad():

        x -= x.grad * learning_rate
        P -= P.grad * learning_rate

        # Manually zero the gradients after updating weights
        x.grad.zero_()
        P.grad.zero_()

print(loss1)
print(loss2)
