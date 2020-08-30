# -*- coding: utf-8 -*-

import Data.Train.trainData as tD

import numpy as np


dt = 0.1


def findVel(xk, xkk):
    return np.sqrt( (xk[0] - xkk[0]) ** 2 + (xk[1] - xkk[1]) **2 ) / dt

def findAngle(xk, xkk):
    return np.arctan( (xkk[1] - xk[1]) / (xkk[0] - xk[0]) )



dataPacks = tD.trainDataPacks

qXs = []
qYs = []
qAngles = []
qVels = []


for dataPack in dataPacks:
    
    for i, (measurementKK, groundTruthKK) in enumerate(zip(dataPack[0], dataPack[1])):
        
        if(i > 0 and i < len(dataPack[0]) - 1):
            
            measurementK, groundTruthK = (dataPack[0][i-1][0], dataPack[1][i-1][0]) 
            measurementKK = measurementKK[0]
            groundTruthKK = groundTruthKK[0]
            measurementKKK, groundTruthKKK = (dataPack[0][i+1][0], dataPack[1][i+1][0])            
            
            
            velK = findVel(groundTruthK, groundTruthKK)
            angleK = findAngle(groundTruthK, groundTruthKK)
            
            velKK = findAngle(groundTruthKK, groundTruthKKK)
            angleKK = findAngle(groundTruthKK, groundTruthKKK)
            
            
            qX = groundTruthKKK[0] - groundTruthKK[0] - velK * dt * np.cos(angleK)
            qY = groundTruthKKK[1] - groundTruthKK[1] - velK * dt * np.sin(angleK)
            
            qAngle = angleKK - angleK
            qVel = velKK - velK
            
            qXs.append(qX)
            qYs.append(qY)
            qAngles.append(qAngle)
            qVels.append(qVel)
            
            
varQx = np.std(qXs) ** 2
varQy = np.std(qYs) ** 2
varQAngle = np.std(qAngles) ** 2
varQVel = np.std(qVels) ** 2

print("varQx : ", varQx)
print("varQy : ", varQy)
print("varQAngle :", varQAngle)
print("varQVel : ",  varQVel)