#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 17:51:39 2020

@author: kuartis
"""


import numpy as np
from scipy.stats import norm, multivariate_normal

import sys
sys.path.append("../")
import commonVariables as common


print_calculateMarginalAssociationProbabilities = False

def print_(*element):
    if(print_calculateMarginalAssociationProbabilities):
        print(element)


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

        eventProbability = calculateJointAssociationProbability_parametric(event, measurements, tracks, spatialDensity, PD)
        sumOfEventProbabilities += eventProbability

        for measurementIndex, trackIndex in enumerate(event):

            marginalAssociationProbabilities[measurementIndex, trackIndex] += eventProbability

    marginalAssociationProbabilities /= sumOfEventProbabilities #normalize the probabilites

    return marginalAssociationProbabilities