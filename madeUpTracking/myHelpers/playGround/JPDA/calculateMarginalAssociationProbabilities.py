#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 17:51:39 2020

@author: kuartis
"""


import numpy as np
from scipy.stats import norm, multivariate_normal

import generateAssociationEvents as gAE
import createValidationMatrix as cVM

from calculateJointAssociationProbability import *

import sys
sys.path.append("../")
import commonVariables as commonVar


print_calculateMarginalAssociationProbabilities = True

def print_(*element):
    if(print_calculateMarginalAssociationProbabilities):
        print(element)


def calculateMarginalAssociationProbabilities(events, measurements, tracks, spatialDensity, PD):

    """
        Description:
            Calculates the marginal association probabilities, ie. Beta(j,t) for each measurement and tracks.
            Note that the value returned from the calculateJointAssociationProbability function is not normalized
            Hence one needs to normali  ze the calculated probabilities.

            calculation : [page 319, (6.2.5-1) BYL95]
            normalization : [page 39, (3-45) AR07]
            
        Input:
            
            events : np.array(shape = (numberOfEvents(not known in advance), m_k, 1))
            
            measurements : np.array(shape = (m,))
            
            tracks : list of len = Nr of track objects
            
            spatialDensity : float 
            
                The poisson parameter that represents the number of false measurements in a unit volume [page 317, (6.2.4-1) BYL95]
                
            "PD" : float(0,1)
            
                Detection probability   
                
        Output:
            
            marginalAssociationProbabilities : np.array(shape = (m_k, Nr))
            
    """
    
    
    numberOfMeasurements = measurements.shape[0]
    numberOfTracks = len(tracks)



    marginalAssociationProbabilities = np.zeros((numberOfMeasurements, numberOfTracks))

    sumOfEventProbabilities = 0 #will be used to normalize the calculated probabilities

    for event in events:

        eventProbability = calculateJointAssociationProbability_parametric(event, measurements, tracks, spatialDensity, PD)
        sumOfEventProbabilities += eventProbability

        for measurementIndex, trackIndex in enumerate(event):
            if(trackIndex != 0):
                marginalAssociationProbabilities[measurementIndex, trackIndex-1] += eventProbability

    marginalAssociationProbabilities /= sumOfEventProbabilities #normalize the probabilites

    return marginalAssociationProbabilities


#playground

marginalAssociationProbabilities = calculateMarginalAssociationProbabilities(gAE.associationEvents, cVM.validatedMeasurements, commonVar.tracks, commonVar.spatialDensity, commonVar.PD)
print_("marginalAssociationProbs : ", marginalAssociationProbabilities)
print("\n",marginalAssociationProbabilities)

