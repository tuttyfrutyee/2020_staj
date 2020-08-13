# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm, multivariate_normal

import sys
sys.path.append("../")
import commonVariables as commonVar


print_updateModeProbabilities = True

def print_(*element):
    if(print_updateModeProbabilities):
        print(element)
        

def updateModeProbabilities(modeStateMeans_measured, modeSs, measurement, transitionMatrix, previousModeProbs): #checkCount : 1
    
    """
        Description : 
            It calculates the new mode probabilities, considering single measurement
        Input:
            modeStateMeans_measured: np.array(shape = (Nr, dimZ))
            modeSs : np.array(shape = (Nr, dimZ, dimZ))
            measurement : np.array(shape = (dimZ, 1))
            transitionMatrix : np.array(shape = (Nr, Nr))
            previousModeProbs : np.array(shape = (Nr,1))

        Output:
            updatedModeProbabilities : np.array(shape = (Nr,1))        

    """


    Nr = modeStateMeans_measured.shape[0]

    likelihoods = np.zeros((Nr,1))

    for i,(modeStateMean_measured, modeS) in enumerate(zip(modeStateMeans_measured, modeSs)):    

        likelihoods[i] = multivariate_normal.pdf(measurement.flatten(), modeStateMean_measured.flatten(), modeS, True)         

    modeSwitch_plainMarkovProbs = np.dot(previousModeProbs.T, transitionMatrix).T
    

    updatedModeProbabilities = modeSwitch_plainMarkovProbs * likelihoods

    #normalize
    updatedModeProbabilities = updatedModeProbabilities / np.sum(updatedModeProbabilities)

    return updatedModeProbabilities


#playground:



updatedModeProbabilities = updateModeProbabilities(commonVar.stateMeans_measured, commonVar.modeSs, commonVar.measurements[0], commonVar.transitionMatrix, commonVar.previousModeProbabilities)
print_(updatedModeProbabilities)
print_(np.sum(updatedModeProbabilities))
