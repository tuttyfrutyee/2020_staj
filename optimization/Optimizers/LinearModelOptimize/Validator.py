# -*- coding: utf-8 -*-

import Tracker
import math
import torch
import sys
sys.path.append("../../Data/Scenarios")
sys.path.append("../../Data/Train")

import matplotlib.pyplot as plt
#%matplotlib qt

import trainData

dtype_torch = torch.float64


def getProcessNoiseCovScale(scale):

    return    torch.tensor([
        [ 0.0251,  0.0219, -0.0072, -0.0054],
        [ 0.0219,  0.0199, -0.0064, -0.0049],
        [-0.0072, -0.0064,  0.0023,  0.0016],
        [-0.0054, -0.0049,  0.0016,  0.0017]
     
    ], dtype=dtype_torch) * scale


def calculateLoss(groundTruth, predictionMeasured):
    
    diff = groundTruth - predictionMeasured
    
    return torch.mm(diff.T, diff)



def createScaleTestPoints(initialScale, finalScale, logStep, linearStepPointCount):
    scalesToTest = []
    
    scale = initialScale
    
    while(True):
        
        nextScale = scale * logStep
        if(nextScale > finalScale):
            nextScale = finalScale
        
        tempScale = scale
        scaleLinearIncrement = (nextScale - scale) / linearStepPointCount
        
        for i in range(linearStepPointCount):
            
            scalesToTest.append(scale)
            scale += scaleLinearIncrement
            
        scale = nextScale
        
        if(scale == finalScale):
            scalesToTest.append(finalScale)
            break

    return scalesToTest


def generateLossSpace(dataPacks, scalesToTest, dt):
    
    lossSpace = []
    
    for i,scale in enumerate(scalesToTest):

        print(str(i / len(scalesToTest) * 100) + "% has been complete")
        
        loss = 0
        
        processNoiseCov = getProcessNoiseCovScale(scale)
        Tracker.ProcessNoiseCov = processNoiseCov
        
        for dataPack in dataPacks:
            
            tracker = Tracker.Tracker_SingleTarget_SingleModel_Linear_allMe()
            measurementPacks, groundTruthPacks = dataPack
            
            for i, (measurementPack, groundTruthPack) in enumerate(zip(measurementPacks, groundTruthPacks)):
                
                z = tracker.feedMeasurement(torch.from_numpy(measurementPack[0]), dt)
                if(z is not None):
                    loss += calculateLoss(torch.from_numpy(groundTruthPack[0]), z) / len(measurementPacks)
        
        loss /= len(dataPacks)
        
        lossSpace.append(loss)
    
            
    return lossSpace



def visualizeLossSpace(scalesToTest, lossSpace, log):
    
    plt.plot(scalesToTest, lossSpace)
    if(log):
        plt.xscale("log")


initialScale = 1e-4
finalScale = 15
logStep = 2
linearStepPointCount = 8
dt = 0.1

scalesToTest = createScaleTestPoints(initialScale, finalScale, logStep, linearStepPointCount)
lossSpace = generateLossSpace(trainData.valDataPacks, scalesToTest, dt)

visualizeLossSpace(scalesToTest[-40:], lossSpace[-40:], False)

                    
        
        
