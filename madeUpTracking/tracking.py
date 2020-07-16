#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:55:25 2020

@author: tuttyfruty
"""

import objectCreator as oC

deltaT = 1
processAccVariance = 10
measurementVariance = 5
iterationCount = 1000

useProcessNoise = True

from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt


def linearLine(x):
    return 3*x*x + 1

def velX(x):
    return 6*x

vz = []
for i in range(iterationCount):
    vz.append(velX(i * deltaT))

z = []
for i in range(iterationCount):
    z.append(linearLine(i * deltaT))
    

noise = np.random.normal(0, np.sqrt(measurementVariance), iterationCount)

#plt.plot(noise)

#plt.plot(z + noise)

measurements = z + noise
#plt.plot(measurements)




from filterpy.common import Q_discrete_white_noise
Q = np.array([ [(deltaT**4)/4., (deltaT**3)/2.],[deltaT**3/2., deltaT**2] ]) * processAccVariance
if( not useProcessNoise) : 
    Q = 0

def createKalmanFilter():
    f = KalmanFilter(dim_x = 2, dim_z = 1)
    f.x = np.array([0.01, 0.1])
    
    f.F = np.array([[1., deltaT, ],
                    [0, 1.]]) 
    
    f.H = np.array([[1., 0.]])    
    f.P *= 10
    f.R = measurementVariance
    f.Q = Q
    
    return f


f = createKalmanFilter()
x_predictions = np.array([])
vx_predictions = np.array([])

for i in range(iterationCount):
    
    f.predict()
    
    f.update(measurements[i])
    
    x_predictions = np.append(x_predictions, [f.x_post[0]])
    vx_predictions = np.append(vx_predictions, [f.x_post[1]])    


#plt.plot(z)
#plt.plot(x_predictions)
#plt.plot(vx_predictions)

#plt.plot(z-x_predictions)

plt.figure()
plt.plot(vz-vx_predictions)
plt.figure()
plt.plot(z-x_predictions)
print(vz[-1] - vx_predictions[-1])
print(z[-1] - x_predictions[-1])

plt.figure()
plt.plot(x_predictions[-20:])
plt.plot(z[-20:])

