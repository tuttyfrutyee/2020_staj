# -*- coding: utf-8 -*-
#%matplotlib qt
import matplotlib.pyplot as plt
import numpy as np
import Scenarios.scenario as scn
import Trackers.SingleTarget.allMe.track_singleTarget_singleModel as allMe_SingleTarget_SingleModel


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


measurementPacks = extractMeasurementsFromScenario(scn.scenario_0)

scn.scenario_0.plotScenario()

predictions = []

dt = 0.1
tracker = allMe_SingleTarget_SingleModel.Tracker_SingleTarget_SingleModel_allMe(1)

measurements = []

for measurementPack in measurementPacks:
    
    #singleTarget
    measurement = measurementPack[0]
    tracker.feedMeasurement(measurement, dt)
    measurements.append(measurement)
    
    if(tracker.track is not None):
        predictions.append(tracker.track.z)

predictions = np.squeeze(predictions)
measurements = np.array(measurements)

plt.plot(measurements[:,0], measurements[:,1])    
plt.plot(predictions[:,0], predictions[:,1], linewidth=2)


































































