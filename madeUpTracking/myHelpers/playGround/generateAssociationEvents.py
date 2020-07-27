#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:49:36 2020

@author: kuartis
"""

import numpy as np
import time

meas1 = [1,1,1]
meas2 = [1,1,1]
meas3 = [1,1,1]
meas4 = [1,1,1]

measurements = [meas1, meas2, meas3, meas4]

validationMatrix = []
for meas in measurements:
    validationMatrix.append(np.array(meas))
validationMatrix = np.array(validationMatrix)




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


events = generateAssociationEvents(validationMatrix)
print(events)