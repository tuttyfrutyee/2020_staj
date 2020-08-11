# -*- coding: utf-8 -*-

import numpy as np


def findCovariance(Rx, Ry, iterationCount):
    
    counter = 0
    
    covRunning = 0
    
    while(counter < iterationCount):
        
        x3 = np.random.normal(0, np.sqrt(Rx))
        x2 = np.random.normal(0, np.sqrt(Rx))
        x1 = np.random.normal(0, np.sqrt(Rx))

        y3 = np.random.normal(0, np.sqrt(Ry))
        y2 = np.random.normal(0, np.sqrt(Ry))
        y1 = np.random.normal(0, np.sqrt(Ry))

    
        x = x3
        y = y3
        phi = np.arctan((y3-y2) / (x3-x2))
        vel = np.sqrt((x3-x2)**2 + (y3-y2)**2)
        phiDot = phi - np.arctan((y2-y1) / (x2-x1))
        
        a = np.array([x,y,phi, vel, phiDot]).reshape((5,1))
        
        cov = np.dot(a,a.T)
        
        covRunning = covRunning * ( counter / (counter+1) ) + cov / (counter + 1)
        
        counter += 1
    
    return covRunning



cov1 = findCovariance(2,2, 1e6)
cov2 = findCovariance(2,2, 1e7)
