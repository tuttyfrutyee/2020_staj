#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:03:53 2020

@author: kuartis
"""
import numpy as np
import sys
sys.path.append("../")
import commonVariables as commonVar


print_createValidationMatrix = True

def print_(*element):
    if(print_createValidationMatrix):
        print(element)

def mahalanobisDistanceSquared(z, mean, cov):

    """
        Description:
            Computes the Mahalanobis distance between the state vector x from the
            Gaussian `mean` with covariance `cov`. This can be thought as the number
            of standard deviations x is from the mean, i.e. a return value of 3 means
            z is 3 std from mean.
        Input:
            z : np.array(shape = (dimZ,1))
            mean : np.array(shape = (dimZ,1))
            cov : np.array(shape = (dimZ,dimZ))
        Output:
            dist : float

    """
    if z.shape != mean.shape:
        raise ValueError("length of input vectors must be the same")

    y = z - mean
    S = np.atleast_2d(cov)

    dist = float(np.dot(np.dot(y.T, np.linalg.pinv(S)), y))
    return dist

def createValidationMatrix(measurements, tracks, gateThreshold):

    """
        Description: 
            This function creates a validation matrix and the indexes of the measurements that are in range

            Inputs:
                measurements : np.array(shape =(m_k, dimZ))
                tracks : list of len = Nr of track objects
                gateThreshold : float(0,1)

            Output:

                validateMeasurementsIndexes: np.array(shape = (numOfSuceesfulGatedMeasurements(not known in advance),1))
                    
                     "The indexes of the measurements which are used to create the validationMatrix. The indexes indicating the
                     element index of that measurement inside the input 'measurements', the order measurements hold should not change"

                    [page 311, Remark, BYL95]

                validationMatrix: np.array(shape = (m_k, Nr+1))

                    The matrix whose shape is ( len(validatedMeasurements), (len(targets) + 1) )
                    The binary elements it contains represents if that column(target) and row(measurement) are in the gating range

                    [page 312, 6.2.2-2, BYL95]
    """

    validationMatrix = []
    validatedMeasurementIndexes = []

    for i,measurement in enumerate(measurements):

        validationVector = [1] # inserting a 1 because -> [page 312, 6.2.2-1, BYL95]
        measurement = np.expand_dims(measurement, axis=1)

        for track in tracks:

            mahalanobisDistanceSquared_ = mahalanobisDistanceSquared(measurement, track.z_prior, track.S)

            if(mahalanobisDistanceSquared_ < gateThreshold):
                validationVector.append(1)
            else:
                validationVector.append(0)

        validationVector = np.array(validationVector)

        if(np.sum(validationVector) > 1): # this means that measurement is validated at least for one track -> worth to consider hence append
            
            validationMatrix.append(validationVector)
            validatedMeasurementIndexes.append(i)

    validationMatrix = np.array(validationMatrix, dtype=int)
    validatedMeasurementIndexes = np.expand_dims(np.array(validatedMeasurementIndexes), axis=1)

    return (validatedMeasurementIndexes, validationMatrix)



validatedMeasurementIndexes, validationMatrix = createValidationMatrix(commonVar.measurements, commonVar.tracks, commonVar.gateThreshold)
if(validatedMeasurementIndexes.shape[0] > 0):
    validatedMeasurements = commonVar.measurements[validatedMeasurementIndexes[:,0]]
else: 
    validatedMeasurements = None
print_("validationMatrix: ", validationMatrix)
print_("validatedMeasurementIndexes : ", validatedMeasurementIndexes[:,0])
print_("validated m_k = ", validatedMeasurements.shape[0])


