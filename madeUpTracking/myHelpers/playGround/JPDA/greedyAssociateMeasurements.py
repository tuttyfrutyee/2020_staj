import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append("../")
import commonVariables as commonVar
import visualizeHelper as vH
import time

#%matplotlib qt

print_generateAssociationEvents = True

def print_(*element):
    if(print_generateAssociationEvents):
        print(element)


def mahalanobisDistanceSquared(x, mean, cov): #checkCount : 1, 1/2*confident

    """
    Computes the Mahalanobis distance between the state vector x from the
    Gaussian `mean` with covariance `cov`. This can be thought as the number
    of standard deviations x is from the mean, i.e. a return value of 3 means
    x is 3 std from mean.

    """

    y = x - mean
    S = np.atleast_2d(cov)

    dist = float(np.dot(np.dot(y.T, np.linalg.inv(S)), y))
    return dist


def greedyAssociateMeasurements(matureTrackers, initTrackers, measurements, gateThreshold, distanceThreshold):

    """
        Input:

            matureTracks : np.array(shape = (Nr_mature,))
            initTracks : np.array(shape = (Nr_init,))
            measurements : np.array(shape = (m_k, dimZ, 1))
            gateThreshold : float
            distanceThreshold : float

        Output:
            
            unMatchedMeasurements : np.array(shape = (not_known, dimZ, 1))
            initTrackerBoundedMeasurements : np.array(shape = (Nr_init, dimZ, 1))
            distanceMatrix : np.array(shape = (Nr_mature, m_k))

    """
    
    Nr_mature = matureTrackers.shape[0]
    Nr_init = initTrackers.shape[0]
    m_k = measurements.shape[0]
    

    #for mature tracks

    distanceMatrix = np.zeros((Nr_mature, m_k))

    distances = []
    distancePairIndexes = []

    for i, tracker in enumerate(matureTrackers):
        for j, measurement in enumerate(measurements):
            
            distanceMatrix[i][j] = mahalanobisDistanceSquared(measurement, tracker.track.z_predict, tracker.track.S)

            distances.append(distanceMatrix[i][j])
            distancePairIndexes.append((i, j))

    sorted_indexes = np.argsort(distances)    

    matchedTrackers = []
    matchedMeasurements = []

    for i in sorted_indexes:
        m, n = distancePairIndexes[i]

        if(m not in matchedTrackers and n not in matchedMeasurements):
            if(distances[i] < gateThreshold):
                matchedTrackers.append(m)
                matchedMeasurements.append(n)

    #for init trackers


    leftMeasurements = []
        
    for i, measurement in enumerate(measurements):
        if(i not in matchedMeasurements):
            leftMeasurements.append(measurement)
    

    distances = []
    distancePairIndexes = []

    for i, tracker in enumerate(initTrackers):
        for j, measurement in enumerate(leftMeasurements):
            diff = tracker.measurements[-1] - measurement
            distances.append(np.sqrt(np.dot(diff.T, diff))[0][0])
            distancePairIndexes.append((i, j))

    sorted_indexes = np.argsort(distances)

    matchedTrackers = []
    matchedMeasurements = []
    
    for i in sorted_indexes:
        m, n = distancePairIndexes[i]

        if(m not in matchedTrackers and n not in matchedMeasurements):
            if(distances[i] < distanceThreshold):
                matchedTrackers.append(m)
                matchedMeasurements.append(n)

    #find unmatched measurements

    unmatchedMeasurements = []
        
    for i, measurement in enumerate(leftMeasurements):
        if(i not in matchedMeasurements):
            unmatchedMeasurements.append(measurement)   

    #find initTrackerBoundedMeasurements

    initTrackerBoundedMeasurements = []

    for m, track in enumerate(initTrackers):
        if(m not in matchedTrackers):
            initTrackerBoundedMeasurements.append(None)
        else:
            initTrackerBoundedMeasurements.append(leftMeasurements[matchedMeasurements[matchedTrackers.index(m)]])

    return (unmatchedMeasurements, initTrackerBoundedMeasurements, distanceMatrix)



#playground
measurements = np.expand_dims(commonVar.measurements, axis=2)
unmatchedMeasurements, initTrackerBoundedMeasurements, distanceMatrix = greedyAssociateMeasurements(commonVar.matureTrackers, commonVar.initTrackers, measurements, commonVar.gateThreshold, commonVar.distanceThreshold)

plt.scatter(measurements[:,0,0], measurements[:,1,0], c="g", linewidths = 15)

for i, matureTracker in enumerate(commonVar.matureTrackers):
    plt.scatter(matureTracker.track.z_predict[0], matureTracker.track.z_predict[1], linewidths = 5, label = "matureTrack - "+str(i))
    vH.showPerimeter(matureTracker.track.z_predict, np.linalg.inv(matureTracker.track.S), np.pi/100, commonVar.gateThreshold)
    
for initTracker in commonVar.initTrackers:
    plt.scatter(initTracker.measurements[-1][0], initTracker.measurements[-1][1])
    vH.showRadius(initTracker.measurements[-1], commonVar.distanceThreshold, np.pi/100)

plt.legend()


print(unmatchedMeasurements)
print(initTrackerBoundedMeasurements)











