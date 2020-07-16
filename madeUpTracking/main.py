# -*- coding: utf-8 -*-

import objectCreator as oC
from track_singleTarget_singleModel import Tracker_SingleTarget_LinearSingleModel

import numpy as np
import matplotlib.pyplot as plt

#%matplotlib qt

plot = False


def calculateL2Error(originalObject, predictionObject):
    
    return np.sum(np.sqrt(  np.square(originalObject[:,0] - predictionObject[:,0]) + \
                            np.square(originalObject[:,1] - predictionObject[:,1])))
    


(originalObjectPaths, noisyObjectPaths) = oC.generateObjectPaths(objectCount = 2, std = 4, pointDistance = 0.5, frame = {"w":200,"h":120})
noisyObject0 = noisyObjectPaths[0]
originalObject0 = originalObjectPaths[0]



# plt.plot(noisyObject0[:,0], noisyObject0[:,1])
# plt.plot(originalObject0[:,0], originalObject0[:,1])



tracker1 = Tracker_SingleTarget_LinearSingleModel(modelType = 0, deltaT = 0.1, measurementNoiseStd = 4)

tracker2 = Tracker_SingleTarget_LinearSingleModel(modelType = 1, deltaT = 0.1, measurementNoiseStd = 4)


for point in noisyObject0:
    tracker1.predictAndUpdate(point)
    tracker2.predictAndUpdate(point)

predictions1 = np.array(tracker1.updatedPredictions).reshape(len(tracker1.updatedPredictions), 2)

predictions2 = np.array(tracker2.updatedPredictions).reshape(len(tracker2.updatedPredictions), 2)




if(plot):
    
    plt.figure()

    plt.plot(predictions1[:,0], predictions1[:,1], 'r', linewidth=2)
    plt.plot(predictions1[-100:,0], predictions1[-100:,1], 'r', linewidth=3.5)
    
    plt.plot(predictions2[:,0], predictions2[:,1], 'k', linewidth=1.5)
    plt.plot(predictions2[-100:,0], predictions2[-100:,1], 'k', linewidth=3.5)
    
    
    plt.plot(originalObject0[:,0], originalObject0[:,1], 'b', linewidth=2)
    plt.plot(originalObject0[-100:,0], originalObject0[-100:,1], 'b', linewidth=3.5)
    
    plt.plot(noisyObject0[:,0], noisyObject0[:,1], "g", linewidth=0.5)
    
    plt.scatter(originalObject0[100,0], originalObject0[100,1], linewidths=10, c='g' )

print(calculateL2Error(originalObject0[100:,:], predictions1[100:,:]))
print(calculateL2Error(originalObject0[100:,:], predictions2[100:,:]))


# plt.plot((predictions - originalObject0)[:,0])
# plt.plot((predictions - originalObject0)[:,1])

# plt.plot(predictions[:,0], predictions[:,1], linestyle='none', marker='o')
