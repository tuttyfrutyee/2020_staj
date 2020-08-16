# -*- coding: utf-8 -*-

import deneme

import Scenarios.scenario as scn


import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim


dtype_torch = torch.float64
dtype_numpy = np.float

torch.autograd.set_detect_anomaly(True)

def extractMomentsFromScenario(scenario):
    momentPacks = []
    i = 0
    done = False
    
    object_ = scenario.objects[0]

    while(i < len(object_.xNoisyPath)):
        
        momentPacks.append( torch.tensor([object_.xNoisyPath[i], object_.yNoisyPath[i], object_.xPath[i], object_.yPath[i]]).reshape((4,1)) )
        i += 1
        
    
    return momentPacks



# for scenario in scn.scenarios_0:
#     scenario.plotScenario()
    
scenarios_moments = []

for scenario in scn.scenarios_0:    
    scenarios_moments.append(extractMomentsFromScenario(scenario))
    
    
learningRate = 1e-3


class GenerateLowerTriangular(torch.autograd.Function):
    @staticmethod
    def forward(self, theta):
        self.save_for_backward(theta)
        
        willReturn = torch.tensor([
                [theta[0], 0,        0,        0],
                [theta[1], theta[2], 0,        0],
                [theta[3], theta[4], theta[5], 0],
                [theta[6], theta[7], theta[8], theta[9]]
            ])
        
        # willReturn = torch.tensor([
        #         [theta[0], 0,        0,          0],
        #         [0,        theta[1], 0,          0],
        #         [0,        0,        theta[2],   0],
        #         [0,        0,        0,          theta[3]]
        #     ])        
        
        
        return willReturn
    
    @staticmethod
    def backward(self, grad_output):
        
        theta, = self.saved_tensors
        
        grad_input = grad_output.clone()
        
        willReturn = torch.zeros(10)
        # willReturn = torch.zeros(4)
        
        #making sure the eigen values stay positive
        
        willReturn[0] = grad_input[0][0] if (theta[0] - learningRate * grad_input[0][0] > 0) else willReturn[0] 
        willReturn[2] = grad_input[1][1] if (theta[2] - learningRate * grad_input[1][1] > 0) else willReturn[2]
        willReturn[5] = grad_input[2][2] if (theta[5] - learningRate * grad_input[2][2] > 0) else willReturn[5]
        willReturn[9] = grad_input[3][3] if (theta[9] - learningRate * grad_input[3][3] > 0) else willReturn[9]
        
        willReturn[1] = grad_input[1][0]
        willReturn[3] = grad_input[2][0]
        willReturn[4] = grad_input[2][1]
        willReturn[6] = grad_input[3][0]
        willReturn[7] = grad_input[3][1]
        willReturn[8] = grad_input[3][2]
        
        #################
        # willReturn[0] = grad_input[0][0] if (theta[0] - learningRate * grad_input[0][0] > 0) else 0
        # willReturn[1] = grad_input[1][1] if (theta[1] - learningRate * grad_input[1][1] > 0) else 0
        # willReturn[2] = grad_input[2][2] if (theta[2] - learningRate * grad_input[2][2] > 0) else 0
        # willReturn[3] = grad_input[3][3] if (theta[3] - learningRate * grad_input[3][3] > 0) else 0        
        
        
        return willReturn    
    
lowerTriangular = GenerateLowerTriangular.apply

Q = 0.005

theta = torch.tensor([
    Q, 0, Q, 0, 0, Q, 0, 0, 0, Q
], dtype= dtype_torch, requires_grad=True)
#theta = torch.rand(10, dtype = dtype_torch, requires_grad = True)
# theta = torch.rand(4, dtype = dtype_torch, requires_grad = True)


optimizer = optim.Adam([theta], lr = learningRate)

triangular = lowerTriangular(theta)
processNoise = torch.mm(triangular.T, triangular)
deneme.processNoise = processNoise


sequenceLength = 100


dt = 0.1

lossClasses = []
for _ in scenarios_moments:
    lossClasses.append([])

for k in range(100000):
    for s, scenario_moments in enumerate(scenarios_moments):
        
        i = 0
        
        tracker = deneme.Tracker_SingleTarget_SingleModel_allMe(0)
        
        
        for l, moment in enumerate(scenario_moments):
            
            if(l == len(scenario_moments) - 1):
                tracker.feedMoment(moment, dt, True)
                
                # with torch.no_grad():
                #     theta -= learningRate * theta.grad
                #     theta.grad.zero_()
                
                optimizer.step()
                optimizer.zero_grad()

                triangular = lowerTriangular(theta)
                processNoise = torch.mm(triangular.T, triangular)
                deneme.processNoise = processNoise 
                                                   
                
            else:
                tracker.feedMoment(moment, dt, False)
        
        lossClasses[s].append(tracker.losses[-1])
                
        
    print("loss_ " + str(k) + " : " , tracker.losses[-1])
               
            
        

plt.plot(lossClasses[0])
plt.plot(lossClasses[1])


a = np.array(processNoise.tolist())








