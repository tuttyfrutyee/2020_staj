# -*- coding: utf-8 -*-
#%matplotlib qt
import matplotlib.pyplot as plt
import numpy as np
import Scenarios.scenario as scn

from Trackers.SingleTarget.allMe.track_singleTarget_singleModel import Tracker_SingleTarget_SingleModel_allMe
from Trackers.SingleTarget.allMe.track_singleTarget_multipleModel import Tracker_SingleTarget_IMultipleModel_allMe

from myHelpers.visualizeHelper import showPerimeter


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


measurementPacks = extractMeasurementsFromScenario(scn.scenario_5)
groundTruthPacks = extractGroundTruthFromScenario(scn.scenario_5)

scn.scenario_5.plotScenario()

predictions = []

dt = 0.1

imm = True
if(not imm):
    tracker = Tracker_SingleTarget_SingleModel_allMe(2)
else:
    tracker = Tracker_SingleTarget_IMultipleModel_allMe()

measurements = []
states = []
modeProbs = []

calLoss = True
loss = 0

S = None
z = None
gateThreshold = 5

eigens = []

for i, (groundTruthPack, measurementPack) in enumerate(zip(groundTruthPacks, measurementPacks)):
    
    #singleTarget
    measurement = measurementPack[0]
    groundTruth = groundTruthPack[0]
    
    tracker.feedMeasurement(measurement, dt)
    measurements.append(measurement)
    
    if(tracker.track is not None):
        predictions.append(tracker.track.z)

        diff = tracker.track.z - groundTruth

        states.append(tracker.track.x)
        
        eigens.append(np.max(np.linalg.eig(tracker.track.P)[0]))
        
        if(not imm and i == len(groundTruthPacks)-1):
            S = tracker.track.S
            z = tracker.track.z_predict
        
        if(calLoss):
            loss += np.sqrt(np.sum(np.dot(diff.T, diff)))
        if(imm):
            modeProbs.append(tracker.modeProbs)

predictions = np.squeeze(predictions)
measurements = np.array(measurements)
states = np.array(states)
if(imm):
    modeProbs = np.array(modeProbs)

plt.plot(measurements[:,0], measurements[:,1])    
plt.plot(predictions[:,0], predictions[:,1], linewidth=2)

#plt.plot(states[:,2,0] / np.pi * 180)
#plt.plot(states[:,3,0])
if(imm):
    plt.figure()
    for i in range(modeProbs.shape[1]):
        plt.plot(modeProbs[:,i], label="model "+str(i)) 

    plt.legend()
    
if(not imm and 0):
    plt.figure()
    plt.plot(eigens[200:])

#showPerimeter(z, np.linalg.inv(S), np.pi / 100, gateThreshold)

























































