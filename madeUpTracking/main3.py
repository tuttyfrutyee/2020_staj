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


measurementPacks = extractMeasurementsFromScenario(scn.scenario_0)
groundTruthPacks = extractGroundTruthFromScenario(scn.scenario_0)

scn.scenario_0.plotScenario()



modelType = 2
gateThreshold = 5
distanceThreshold = 5
spatialDensity = 0.05
detThreshold = 200
PD = 0.999

dt = 0.1


multipleTargetTracker = Tracker_MultipleTarget_SingleModel_allMe(modelType, gateThreshold, distanceThreshold, detThreshold, spatialDensity, PD)


for i, measurementPack in enumerate(measurementPacks):

    measurements = np.array(measurementPack)
    multipleTargetTracker.feedMeasurements(measurements, dt, i)
    


    
        
        
# for initTracker in multipleTargetTracker.initTrackerHistory:
      
#     showRadius(initTracker.measurements[-1], distanceThreshold, np.pi/100)
                
        
print("validationMatrix.shape = ", multipleTargetTracker.validationMatrix.shape)
print("associationEvents.shape = ", multipleTargetTracker.associationEvents.shape)


# ani = visualizeTrackingResults(multipleTargetTracker.matureTrackerHistory, measurementPacks, groundTruthPacks, True, gateThreshold)

ani = visualizeTrackingResults(multipleTargetTracker.matureTrackerHistory, measurementPacks, groundTruthPacks, False, gateThreshold)
        
        
        