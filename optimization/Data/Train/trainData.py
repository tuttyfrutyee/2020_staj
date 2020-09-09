# -*- coding: utf-8 -*-

import sys
sys.path.append("./Data")
sys.path.append("./Data/Scenarios")
import scenarioGenerator as scenario

import numpy as np

def extractMeasurementsFromScenario(scenario):
    measurementPacks = []
    i = 0
    exhausteds = np.zeros((len(scenario.objects)))
    done = False

    while(not done):
        measurementPack = []
        for k,object_ in enumerate(scenario.objects):

            if(len(object_.xPath) > i):
                measurementPack.append(np.expand_dims(np.array([object_.xNoisyPath[i], object_.yNoisyPath[i]]), axis=1))
            else:
                exhausteds[k] = 1
        if(np.sum(exhausteds) > len(scenario.objects)-1):
            done = True
            
        if(not done):
            measurementPacks.append(measurementPack)
        i += 1
            
    return measurementPacks

def extractGroundTruthFromScenario(scenario):
    groundTruthPacks = []
    i = 0
    exhausteds = np.zeros((len(scenario.objects)))
    done = False

    while(not done):
        groundTruthPack = []
        for k,object_ in enumerate(scenario.objects):

            if(len(object_.xPath) > i):
                groundTruthPack.append(np.expand_dims(np.array([object_.xPath[i], object_.yPath[i]]), axis=1))
            else:
                exhausteds[k] = 1
        if(np.sum(exhausteds) > len(scenario.objects)-1):
            done = True
            
        if(not done):
            groundTruthPacks.append(groundTruthPack)
        i += 1

    return groundTruthPacks



numberOfScenario = 30
std = 1.2
plotCount = 0

scenarios = []
dataPacks = []

trainDataPacks = []
valDataPacks = []

trainRatio = 0.9

for s in range(numberOfScenario):
    
    stds = [std]
    objectPathCorners = [ None ] 
    corruptions = [None]
    stepSizes = [0.4]
    colors = [("b", "g")]
    
    scenario_ = scenario.Scenario(stds, objectPathCorners, corruptions, stepSizes, colors)
    
    scenarios.append(scenario)
    
    if(s < plotCount):
        scenario_.plotScenario()
    
    measurementPacks = extractMeasurementsFromScenario(scenario_)
    groundTruthPacks = extractGroundTruthFromScenario(scenario_)
    
    if(s < numberOfScenario * trainRatio):
        trainDataPacks.append((measurementPacks, groundTruthPacks))
    else:
        valDataPacks.append((measurementPacks, groundTruthPacks))
    
    
    
    
    
    
    
    
    
    
    
    