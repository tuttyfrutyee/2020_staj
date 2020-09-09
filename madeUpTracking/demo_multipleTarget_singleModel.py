import matplotlib.pyplot as plt
import numpy as np
import Scenarios.scenario as scn
import copy
import time

from Trackers.MultipleTarget.allMe.track_multipleTarget_singleModel import Tracker_MultipleTarget_SingleModel_allMe

#visualizer helpers

from myHelpers.visualizeHelper import showPerimeter
from myHelpers.visualizeHelper import showRadius
from myHelpers.visualizeHelper import visualizeTrackingResults


#%matplotlib qt

cutMeasurementIndex = float("inf")


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
                    if(i < cutMeasurementIndex):
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

measurementPacks = extractMeasurementsFromScenario(scn.scenario_2)
groundTruthPacks = extractGroundTruthFromScenario(scn.scenario_2)


#parameters

modelType = 1
gateThreshold = 10
distanceThreshold = 5
spatialDensity = 0.0001
detThreshold = 200
PD = 0.99
dt = 0.1

# get the tracker
multipleTargetTracker = Tracker_MultipleTarget_SingleModel_allMe(modelType, gateThreshold, distanceThreshold, detThreshold, spatialDensity, PD)

start = time.time()
# tracking happens here
for i, measurementPack in enumerate(measurementPacks):
    print(str(i / len(measurementPacks) * 100) + "% complete")
    measurements = np.array(measurementPack)
    multipleTargetTracker.feedMeasurements(measurements, dt, i)

print("multipleTarget_singleModel fps : ", i/(time.time() - start), "with " + str(len(scn.scenario_3.objects)) + " number of tracks")
fps = i/(time.time() - start)

#plotting

scn.scenario_2.plotScenario()

# ani = visualizeTrackingResults(multipleTargetTracker.matureTrackerHistory, measurementPacks, groundTruthPacks, True, gateThreshold)

ani = visualizeTrackingResults(multipleTargetTracker.matureTrackerHistory, measurementPacks, groundTruthPacks, False, gateThreshold)
        
        
        