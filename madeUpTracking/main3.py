import matplotlib.pyplot as plt
import numpy as np
import Scenarios.scenario as scn
import copy

from Trackers.MultipleTarget.allMe.track_multipleTarget_singleModel import Tracker_MultipleTarget_SingleModel_allMe

from myHelpers.visualizeHelper import showPerimeter
from myHelpers.visualizeHelper import showRadius
from myHelpers.visualizeHelper import visualizeTrackingResults


#%matplotlib qt
#scn.scenario_2.plotScenario()


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
    
    for k, object_ in enumerate(scenario.objects):
        
        groundTruthX = object_.xPath
        groundTruthY = object_.yPath
        
        groundTruthPacks.append((groundTruthX, groundTruthY))
    
    return groundTruthPacks


measurementPacks = extractMeasurementsFromScenario(scn.scenario_2)
groundTruthPacks = extractGroundTruthFromScenario(scn.scenario_2)

scn.scenario_2.plotScenario()



modelType = 1
gateThreshold = 5
distanceThreshold = 7
spatialDensity = 0.4
PD = 0.99

dt = 0.1


multipleTargetTracker = Tracker_MultipleTarget_SingleModel_allMe(modelType, gateThreshold, distanceThreshold, spatialDensity, PD)

snapshot = None

for i, measurementPack in enumerate(measurementPacks):

    measurements = np.array(measurementPack)
    multipleTargetTracker.feedMeasurements(measurements, dt, i)
    
    predictions = []
    for tracker in multipleTargetTracker.matureTrackers:
        predictions.append(tracker.track.z)
    
    if(i == 5):
        snapshot = copy.deepcopy(multipleTargetTracker)
    
    


    
        
        
# for initTracker in multipleTargetTracker.initTrackerHistory:
      
#     showRadius(initTracker.measurements[-1], distanceThreshold, np.pi/100)
                
        
print("validationMatrix.shape = ", multipleTargetTracker.validationMatrix.shape)
print("associationEvents.shape = ", multipleTargetTracker.associationEvents.shape)



ani = visualizeTrackingResults(multipleTargetTracker.matureTrackerHistory, measurementPacks, groundTruthPacks, False, gateThreshold)
        
        
        