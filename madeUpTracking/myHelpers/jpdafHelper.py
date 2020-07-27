from scipy.stats import norm, multivariate_normal
import numpy as np
import scipy.linalg as linalg
from numpy.linalg import inv

import math


"""
    References:

    [BYL95] : MultiTarget-Multisensor Tracking Yaakov Bar-Shalom 1995
    [AR07] : Masters Thesis: 3D-LIDAR Multi Object Tracking for Autonomous Driving

"""

def generateAssociationEvents(validationMatrix):

    events = []

    numberOfMeasurements = validationMatrix.shape[0]
    exhaustedMeasurements = np.zeros((numberOfMeasurements), dtype=int)

    usedTrackers = None
    previousEvent = np.zeros(shape = (numberOfMeasurements), dtype = int) - 1
    burnCurrentEvent = None

    while(not exhaustedMeasurements[0]):

        event = np.zeros(shape = (numberOfMeasurements), dtype=int)
        burnCurrentEvent = False
        usedTrackers = []

        for i,validationVector in enumerate(validationMatrix):
            

            if(previousEvent[i] == -1):
                event[i] = 0
            else:

                nextMeasurementIndex = i+1
                if(nextMeasurementIndex == numberOfMeasurements or exhaustedMeasurements[nextMeasurementIndex]):

                    if(nextMeasurementIndex != numberOfMeasurements):
                        exhaustedMeasurements[nextMeasurementIndex:] = 0
                        previousEvent[nextMeasurementIndex:] = -1

                    
                    nextTrackIndex = previousEvent[i]
                    
                    
                    while(validationVector.shape[0]-1 > nextTrackIndex):
                        if(nextTrackIndex != previousEvent[i]):
                            if((nextTrackIndex not in usedTrackers) and (validationVector[nextTrackIndex])):
                                break
                        nextTrackIndex +=1
                        
                    if(not validationVector[nextTrackIndex] or nextTrackIndex == previousEvent[i] or nextTrackIndex in usedTrackers):
                        burnCurrentEvent = True
                        exhaustedMeasurements[i] = 1
                        break
                    

                    usedTrackers.append(nextTrackIndex)
                    event[i] = nextTrackIndex
                
                else:
                    event[i] = previousEvent[i]
                    usedTrackers.append(previousEvent[i])
            
        if(burnCurrentEvent):
            


            continue

        previousEvent = np.copy(event)
        

        events.append(event)
    
    return events

def createValidationMatrix(measurements, tracks, gateThreshold):

    """
        Description: 
            This function creates a validation matrix and the indexes of the measurements that are in range

            (allMeasurements, tracks) --> [Function Box] --> (validateMeasurementsIndexes, validationMatrix)

            Inputs:

                'gateThreshold': 

                    The threshold for gating

                [page 130, 3.4.1-4, BYL95] 

            Returns:

                'validateMeasurementsIndexes' :
                
                     The indexes of the measurements which are used to create the validationMatrix. The indexes indicating the
                element index of that measurement inside the input 'measurements', the order measurements hold should not change

                [page 311, Remark, BYL95]

                'validationMatrix' :

                    The matrix whose shape is ( len(validatedMeasurements), (len(targets) + 1) )
                    The binary elements it contains represents if that column(target) and row(measurement) are in the gating range

                [page 312, 6.2.2-2, BYL95]
    """

    validationMatrix = []
    validatedMeasurementIndexes = []

    for i,measurement in enumerate(measurements):

        validationVector = [1] # inserting a 1 because -> [page 312, 6.2.2-1, BYL95]

        for track in tracks:

            mahalanobisDistanceSquared = mahalanobisDistanceSquared(measurement, track.z_prior, track.S)

            if(mahalanobisDistanceSquared < gateThreshold):
                validationVector.append(1)
            else:
                validationVector.append(0)

        validationVector = np.array(validationVector)

        if(np.sum(validationVector) > 1): # this means that measurement is validated at least for one track -> worth to consider hence append
            
            validationMatrix.append(validationVector)
            validatedMeasurementIndexes.append(i)

    validationMatrix = np.array(validationMatrix)
    validatedMeasurementIndexes = np.array(validatedMeasurementIndexes)

    return (validatedMeasurementIndexes, validationMatrix)



def mahalanobisDistanceSquared(x, mean, cov):

    """
    Computes the Mahalanobis distance between the state vector x from the
    Gaussian `mean` with covariance `cov`. This can be thought as the number
    of standard deviations x is from the mean, i.e. a return value of 3 means
    x is 3 std from mean.

    """
    if x.shape != mean.shape:
        raise ValueError("length of input vectors must be the same")

    y = x - mean
    S = np.atleast_2d(cov)

    dist = float(np.dot(np.dot(y.T, inv(S)), y))
    return dist


def calculateJointAssociationProbability(jAE, measurements, tracks, spatialDensity, PD):

    """
        Description:
            This function calculates the possibility of the association event(jAE) occurrence
            using the parametric JPDA

            Note that, the value returned from this function is not exact, the normalization constant is undetermined
            One needs to divide this returned value with the sum of all joint association probabilities to normalize

            [page 317, (6.2.4-4) BYL95]

        'measurements' : numpy array of shape (m,)

        'tracks' : numpy array of shape (Nt,)
                                                                                   
        'jAE' : "jointAssociationEvent" :

            It is a matrix of shape ( MeasurementsInSurvelience, NumberOfTracks+1 )
            The elements of the matrix show if corresponding row(measurement) and column(tracker) are associated
            jAE = [w^jt]    
            
            [page 312, (6.2.2-3) BYL95]
    
        'spatialDensity': 

            The poisson parameter that represents the number of false measurements in a unit volume

            [page 317, (6.2.4-1) BYL95]

        "PD" :

            Detection probability

    """

    numberOfDetections = np.sum(jAE>0)

    measurementProbabilities = 1

    for measurementIndex, associatedTrack in enumerate(jAE):

        trackPriorMean = tracks[associatedTrack].z_priorMean
        trackS = tracks[associatedTrack].S

        measurementProbabilities *= calculateTheProbabilityOfTheMeasurementRelatedWithTrack(measurements[measurementIndex], trackPriorMean, trackS)
        measurementProbabilities /= spatialDensity

    return measurementProbabilities * pow(PD, numberOfDetections) * pow(1-PD, tracks.shape[0] - numberOfDetections)



def calculateTheProbabilityOfTheMeasurementRelatedWithTrack(measurement, trackPriorMean, trackS):

    """

        Description:
            Calculate the probability that 'measurement' is measured w.r.t target, whos mean(targetPriorMean), innovationCovariance(targetPriorS)
        
        [page 314, (6.2.3-4) BYL95]

    """

    if trackPriorMean is not None:
        flat_targetPriorMean = np.asarray(trackPriorMean).flatten()
    else:
        flat_targetPriorMean = None

    flat_measurement = np.asarray(measurement).flatten()


    return multivariate_normal.pdf(flat_measurement, flat_targetPriorMean, trackS, True)



def calculateMarginalAssociationProbabilities(events, measurements, tracks, spatialDensity, PD):

    """
        Description:
            Calculates the marginal association probabilities, ie. Beta(j,t) for each measurement and tracks.
            Note that the value returned from the calculateJointAssociationProbability function is not normalized
            Hence one needs to normalize the calculated probabilities.

            calculation : [page 317, (6.2.5-1) BYL95]
            normalization : [page 39, (3-45) AR07]

        'measurements' : numpy array of shape (m,)

        'tracks' : numpy array of shape (Nt,)
    
        'spatialDensity': 

            The poisson parameter that represents the number of false measurements in a unit volume

            [page 317, (6.2.4-1) BYL95]

        "PD" :

            Detection probability            

    """
    
    numberOfMeasurements = measurements.shape[0]
    numberOfTracks = tracks.shape[0]

    marginalAssociationProbabilities = np.zeros((numberOfMeasurements, numberOfTracks))

    sumOfEventProbabilities = 0 #will be used to normalize the calculated probabilities

    for event in events:

        eventProbability = calculateJointAssociationProbability(event, measurements, tracks, spatialDensity, PD)
        sumOfEventProbabilities += eventProbability

        for measurementIndex, trackIndex in enumerate(event):

            marginalAssociationProbabilities[measurementIndex, trackIndex] += eventProbability

    marginalAssociationProbabilities /= sumOfEventProbabilities #normalize the probabilites

    return marginalAssociationProbabilities