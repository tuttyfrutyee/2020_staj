# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import Scenarios.scenario as scn
import copy
import time

from Trackers.MultipleTarget.allMe.track_multipleTarget_multipleModel import Tracker_MultipleTarget_MultipleModel_allMe

from myHelpers.visualizeHelper import showPerimeter
from myHelpers.visualizeHelper import showRadius
from myHelpers.visualizeHelper import visualizeTrackingResults


#functions for getting data out of scenarios

def extractMeasurementsFromScenario(scenario):
    measurementPacks = []
    i = 0
    exhausteds = np.zeros((len(scenario.objects)))
    done = False

    while(not done):
        measurementPack = []
        for k,object_ in enumerate(scenario.objects):

            if(len(object_.xPath) > i):
                if(object_.xNoisyPath[i] is not None):
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
    
    for k, object_ in enumerate(scenario.objects):
        
        groundTruthX = object_.xPath
        groundTruthY = object_.yPath
        
        groundTruthPacks.append((groundTruthX, groundTruthY))
    
    return groundTruthPacks



# get data
measurementPacks = extractMeasurementsFromScenario(scn.scenario_3)
groundTruthPacks = extractGroundTruthFromScenario(scn.scenario_3)



#parameters
gateThreshold = 20
distanceThreshold = 5
spatialDensity = 0.00008
detThreshold = 1000
PD = 0.999
dt = 0.1


#get the tracker

multipleTargetTracker = Tracker_MultipleTarget_MultipleModel_allMe(gateThreshold, distanceThreshold, detThreshold, spatialDensity, PD)

#tracking happens here

start = time.time()
for i, measurementPack in enumerate(measurementPacks):

    measurements = np.array(measurementPack)
    multipleTargetTracker.feedMeasurements(measurements, dt, i)
print("fps : ", i / (time.time() - start) )


#plotting

scn.scenario_3.plotScenario()        

# ani = visualizeTrackingResults(multipleTargetTracker.matureTrackerHistory, measurementPacks, groundTruthPacks, True, gateThreshold)
    
ani = visualizeTrackingResults(multipleTargetTracker.matureTrackerHistory, measurementPacks, groundTruthPacks, False, gateThreshold)
        
        

