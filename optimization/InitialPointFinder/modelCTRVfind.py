# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import Data.Train.trainData as tD

import numpy as np

dt = 0.1


def findVel(xk, xkk):
    return np.sqrt( (xk[0] - xkk[0]) ** 2 + (xk[1] - xkk[1]) **2 ) / dt

def findAngle(xk, xkk):
    return np.arctan( (xkk[1] - xk[1]) / (xkk[0] - xk[0]) )

def findAngleVel(xk, xkk, xkkk):
    return findAngle(xkk, xkkk) - findAngle(xk, xkk) / dt


dataPacks = tD.trainDataPacks

qXs = []
qYs = []
qAngles = []
qVels = []
qAngleVels = []


for dataPack in dataPacks:
    
    for i, (measurementKK, groundTruthKK) in enumerate(zip(dataPack[0], dataPack[1])):
        
        if(i > 1 and i < len(dataPack[0]) - 1):
            
            measurement, groundTruth = (dataPack[0][i-2][0], dataPack[1][i-2][0])
            measurementK, groundTruthK = (dataPack[0][i-1][0], dataPack[1][i-1][0]) 
            measurementKK = measurementKK[0]
            groundTruthKK = groundTruthKK[0]
            measurementKKK, groundTruthKKK = (dataPack[0][i+1][0], dataPack[1][i+1][0])            
            
            
            velK = findVel(groundTruthK, groundTruthKK)
            angleK = findAngle(groundTruthK, groundTruthKK)
            
            velKK = findAngle(groundTruthKK, groundTruthKKK)
            angleKK = findAngle(groundTruthKK, groundTruthKKK)
            
            angleVelK = findAngleVel(groundTruth, groundTruthK, groundTruthKK)
            angleVelKK = findAngleVel(groundTruthK, groundTruthKK, groundTruthKKK)
            
            
            qX = groundTruthKKK[0] - groundTruthKK[0] - velK / angleVelK * (np.sin(angleK) - np.sin(angleVelK * dt + angleK))
            qY = groundTruthKKK[1] - groundTruthKK[1] - velK / angleVelK * (-np.cos(angleK) + np.cos(angleVelK * dt + angleK))
            
            qAngle = angleKK - angleK - angleVelK * dt
            qVel = velKK - velK            
            qAngleVel = angleVelKK - angleVelK
            
            qXs.append(qX)
            qYs.append(qY)
            qAngles.append(qAngle)
            qVels.append(qVel)
            qAngleVels.append(qAngleVel)
            
varQx = np.std(qXs) ** 2
varQy = np.std(qYs) ** 2
varQAngle = np.std(qAngles) ** 2    
varQVel = np.std(qVels) ** 2
varQAngleVel = np.std(qAngleVels) ** 2

print("varQx : ", varQx)
print("varQy : ", varQy)
print("varQAngle :", varQAngle)
print("varQVel : ",  varQVel)
print("varQAngleVel : ",  varQAngleVel)
