# -*- coding: utf-8 -*-
#%matplotlib qt
import matplotlib.pyplot as plt
import numpy as np
import Scenarios.scenario as scn

from Trackers.SingleTarget.allMe.track_singleTarget_singleModel import Tracker_SingleTarget_SingleModel_allMe
from Trackers.SingleTarget.allMe.track_singleTarget_multipleModel import Tracker_SingleTarget_IMultipleModel_allMe

from myHelpers.visualizeHelper import showPerimeter


# some helper functions


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


# setting parameters and data


    #get data ready
measurementPacks = extractMeasurementsFromScenario(scn.scenario_0)
groundTruthPacks = extractGroundTruthFromScenario(scn.scenario_0)


loss = 0
dt = 0.1
S = None
z = None
imm = False
modeType = 1


if(not imm):
    tracker = Tracker_SingleTarget_SingleModel_allMe(modeType)
else:
    tracker = Tracker_SingleTarget_IMultipleModel_allMe()

measurements = []
states = []
modeProbs = []
predictions = []




# tracking

for i, (groundTruthPack, measurementPack) in enumerate(zip(groundTruthPacks, measurementPacks)):
    
    #since singleTarget, index 0
    measurement = measurementPack[0]
    groundTruth = groundTruthPack[0]
    
    tracker.feedMeasurement(measurement, dt)
    measurements.append(measurement)
    
    if(tracker.track is not None):
        predictions.append(tracker.track.z)

        diff = tracker.track.z - groundTruth
        loss += np.sqrt(np.sum(np.dot(diff.T, diff)))
      

        states.append(tracker.track.x)
        
        if(not imm and i == len(groundTruthPacks)-1):
            S = tracker.track.S
            z = tracker.track.z_predict
        
        
        if(imm):
            modeProbs.append(tracker.modeProbs)

predictions = np.squeeze(predictions)
measurements = np.array(measurements)
states = np.array(states)


# plotting

scn.scenario_0.plotScenario()


plt.plot(measurements[:,0], measurements[:,1])    
plt.plot(predictions[:,0], predictions[:,1], linewidth=2)


if(imm):
    modeProbs = np.array(modeProbs)    
    plt.figure()
    for i in range(modeProbs.shape[1]):
        plt.plot(modeProbs[:,i], label="model "+str(i)) 

    plt.legend()
    























































