from scipy.stats import norm, multivariate_normal
import numpy as np
import scipy.linalg as linalg
from numpy.linalg import inv

import math


"""
    Index:

    [BYL95] : MultiTarget-Multisensor Tracking Yaakov Bar-Shalom 1995

"""

def generateAssociationEvents(validationMatrix):

    events = []

    for validationVector in validationMatrix:
        


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


def calculateJointAssociationProbability(jAE, spatialDensity, PD):

    """
        Description:
            This function calculates the possibility of the association event(jAE) occurrence
            using the parametric JPDA

            Note that, the value returned from this function is not exact, the normalization constant is undetermined
            One needs to divide this returned value with the sum of all joint association probabilities to normalize

            [page 317, (6.2.4-4) BYL95]
                                                                                   
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

    print("todo: calculateJointAssociationProbability")

def calculateTargetDetectionIndicatorVector(jAE):

    """
        Description:
            Returns a vector such that each element of it shows if that index related target is detected

            [page 313, (6.2.2-6) BYL95]

        'jAE' : "jointAssociationEvent" :

            It is a matrix of shape ( MeasurementsInSurvelience, NumberOfTracks+1 )
            The elements of the matrix show if corresponding row(measurement) and column(tracker) are associated
            jAE = [w^jt]    
            
            [page 312, (6.2.2-3) BYL95]        

    """

    print("todo : calculateTargetDetectionIndicatorVector")


def calculateMeasurementAssociationIndicatorVector(jAE):

    """
        Description:
            Returns a vector such that each element of it shows if that index related measurement is associated with a target

            [page 313, (6.2.2-7) BYL95]

        'jAE' : "jointAssociationEvent" :

            It is a matrix of shape ( MeasurementsInSurvelience, NumberOfTracks+1 )
            The elements of the matrix show if corresponding row(measurement) and column(tracker) are associated
            jAE = [w^jt]    
            
            [page 312, (6.2.2-3) BYL95]        

    """

    print("todo : calculateMeasurementAssociationIndicatorVector")




def calculateTheProbabilityOfTheMeasurementRelatedWithTarget(measurement, targetPriorMean, targetPriorS):

    """

        Description:
            Calculate the probability that 'measurement' is measured w.r.t target, whos mean(targetPriorMean), innovationCovariance(targetPriorS)
        
        [page 314, (6.2.3-4) BYL95]

    """

    if targetPriorMean is not None:
        flat_targetPriorMean = np.asarray(targetPriorMean).flatten()
    else:
        flat_targetPriorMean = None

    flat_measurement = np.asarray(measurement).flatten()


    return multivariate_normal.pdf(flat_measurement, flat_targetPriorMean, targetPriorS, True)