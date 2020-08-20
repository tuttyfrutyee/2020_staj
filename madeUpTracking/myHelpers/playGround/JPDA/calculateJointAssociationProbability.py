#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:50:01 2020

@author: kuartis
"""


import numpy as np
from scipy.stats import norm, multivariate_normal

import generateAssociationEvents as gAE

import sys
sys.path.append("../")
import commonVariables as commonVar
import common as common


print_calculateAssociationProbability = False

def print_(*element):
    if(print_calculateAssociationProbability):
        print(element)
        
        
        
def calculateTheProbabilityOfTheMeasurement(measurement, z, S):

    """

        Description:
            Calculate the probability that 'measurement' is measured w.r.t z and S
        
        [page 314, (6.2.3-4) BYL95]
        
        Input:
            measurement : np.array(shape=(dimZ, 1))
            z : np.array(shape = (dimZ, 1))
            S : np.array(shape = (dimZ, dimZ))
        Output:
            probability : float(0,1)

    """

    return multivariate_normal.pdf(measurement.flatten(), z.flatten(), S, True)        
        
        
def calculateJointAssociationProbability_parametric(jAE, measurements, tracks, spatialDensity, PD): #checkCount : 1

    """
        Description:
            This function calculates the possibility of the association event(jAE) occurrence
            using the parametric JPDA

            Note that, the value returned from this function is not exact, the normalization constant is undetermined
            One needs to divide this returned value with the sum of all joint association probabilities to normalize

            [page 317, (6.2.4-4) BYL95]
        
        Input:
 
            jAE : np.array(shape : (m_k, 1))
                "The elements of the vector represents the targetIndexes the measurements are associated with" [page 312, (6.2.2-3) BYL95]  
               
            measurements : np.array(shape=(m_k,1))
               
            tracks: list of len = Nr of track objects
    
            spatialDensity : float
               "The poisson parameter that represents the number of false measurements in a unit volume" [page 317, (6.2.4-1) BYL95]
    
            PD: float
               "Detection probability"           
            
            Output:
                associationProbability : float  
    """

    numberOfDetections = np.sum(jAE>0)

    measurementProbabilities = 1

    for measurementIndex, associatedTrack in enumerate(jAE):
        
        if(associatedTrack != 0):

            trackPriorMean = tracks[associatedTrack-1].z_prior
            trackS = tracks[associatedTrack-1].S
    
            measurementProbabilities *= calculateTheProbabilityOfTheMeasurement(measurements[measurementIndex], trackPriorMean, trackS)
            measurementProbabilities /= spatialDensity

    return measurementProbabilities * pow(PD, numberOfDetections) * pow(1-PD, len(tracks) - numberOfDetections)


def calculateJointAssociationProbability_nonParametric(jAE, measurements, tracks, volume, PD): #checkCount : 1

    """
        Description:
            This function calculates the possibility of the association event(jAE) occurrence
            using the non parametric JPDA

            Note that, the value returned from this function is not exact, the normalization constant is undetermined
            One needs to divide this returned value with the sum of all joint association probabilities to normalize

            [page 318, (6.2.4-8) BYL95]
        
        Input:
 
            jAE : np.array(shape : (m_k, 1))
                "The elements of the vector represents the targetIndexes the measurements are associated with" [page 312, (6.2.2-3) BYL95]  
               
            measurements : np.array(shape=(m_k,1))
               
            tracks: list of len = Nr of track objects
    
            volume : float
               "volume of the surveillance region" [page 314, (6.2.3-5) BYL95]
    
            PD: float
               "Detection probability"           
        
        Output:
            
            associationProbability : float
    """

    numberOfDetections = np.sum(jAE>0)
    m_k = jAE.shape[0]

    measurementProbabilities = 1
    
    def fact(n):
        a = 1
        for i in range(n):
            a *= (i+1)
        return a

    for measurementIndex, associatedTrack in enumerate(jAE):
        
        if(associatedTrack != 0):

            trackPriorMean = tracks[associatedTrack-1].z_prior
            trackS = tracks[associatedTrack-1].S
    
            measurementProbabilities *= calculateTheProbabilityOfTheMeasurement(measurements[measurementIndex], trackPriorMean, trackS)
            measurementProbabilities *= volume

    return measurementProbabilities * fact(m_k - numberOfDetections) * pow(PD, numberOfDetections) * pow(1-PD, len(tracks) - numberOfDetections)



#playground

jAE = gAE.associationEvents[-1]
print_(jAE)

assocProb_parametric = calculateJointAssociationProbability_parametric(jAE, commonVar.measurements, commonVar.tracks, commonVar.spatialDensity, commonVar.PD)
assocProb_nonParametric = calculateJointAssociationProbability_nonParametric(jAE, commonVar.measurements, commonVar.tracks, common.gatedVolume, commonVar.PD)

print_("assocProb_parametric", assocProb_parametric)
print_("assocProb_nonParametric", assocProb_nonParametric)

# Note  that it is okey that assocProb_parametric != assocProb_nonParametric, they are unnormalized
# Further it is also okey that their normalized outputs are not same, since it all depends what
# have selected as spatialDensity, and what we have used as surveillance volume.


