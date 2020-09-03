#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:49:36 2020

@author: kuartis
"""

import numpy as np
import sys

import createValidationMatrix as cVM


sys.path.append("../")
import commonVariables as commonVar

import time


print_generateAssociationEvents = True

def print_(*element):
    if(print_generateAssociationEvents):
        print(element)


def generateAssociationEvents(validationMatrix):
    
    """
        Description:
            ---
        Input:
            validationMatrix: np.array(shape = (m_k, Nr+1))
        Output:
            associationEvents : np.array(shape = (numberOfEvents(not known in advance), m_k, 1))
    """
    
    associationEvents = []

    m_k = validationMatrix.shape[0]
    exhaustedMeasurements = np.zeros((m_k), dtype=int)

    usedTrackers = None
    previousEvent = np.zeros(shape = (m_k), dtype = int) - 1
    burnCurrentEvent = None
    
    if(m_k == 0):
        return None

    while(not exhaustedMeasurements[0]):

        event = np.zeros(shape = (m_k), dtype=int)
        burnCurrentEvent = False
        usedTrackers = []

        for i,validationVector in enumerate(validationMatrix):
            
            #Note: validationVector corresponds to a measurement -> measurement_validationVector : [val?_t0, val?_t1, ..., val?_tNr]

            if(previousEvent[i] == -1):
                event[i] = 0 #first t_0 will be considered, note it always a choice since measurement can be not related with any track
            else:

                nextMeasurementIndex = i+1
                if(nextMeasurementIndex == m_k or exhaustedMeasurements[nextMeasurementIndex]):

                    if(nextMeasurementIndex != m_k):
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
        associationEvents.append(event)
    
    return np.array(associationEvents, dtype=int)

customValMatrix = np.array([[1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 1]])

start = time.time()
# associationEvents = generateAssociationEvents(cVM.validationMatrix)
associationEvents = generateAssociationEvents(customValMatrix)

print(cVM.validationMatrix)
print(associationEvents)
# associationEvents = generateAssociationEvents(np.ones((5,6)))

print("Took : ", time.time() - start, " seconds")
# print_("associationEvents : ", associationEvents)