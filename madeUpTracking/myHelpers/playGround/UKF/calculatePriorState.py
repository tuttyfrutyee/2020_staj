from scipy.linalg import cholesky
import numpy as np

import generateSigmaPoints as gSP
import generateUnscentedWeights as gUW

import sys
sys.path.append("../")
import commonVariables as commonVar

print_calculatePriorState = True

def print_(*element):
    if(print_calculatePriorState):
        print(element)


def calculatePredictedState(forwardFunc, measureFunc, sigmaPoints, Ws, Wc, processNoise, measurementNoise):

    """
    
        Description : 
            Calculates the predicted state
            [WR00 Algorithm 3.1]
    
        Input:
            forwardFunc: function
            measureFunc: function
            sigmaPoints : np.array(shape = (2*dimX+1,dimX,1))
            Ws : [Ws0, Wsi] -> only 2 elements
            Wc : [Wc0, Wci] -> only 2 elements
            processNoise : np.array(shape = (dimX, dimX))
            measurementNoise : np.array(shape = (dimZ, dimZ))
        
        Output:
            predictedStateMean : np.array(shape = (dimX,1))
            predictedStateCovariance : np.array(shape = (dimX, dimX))
            Pzz : np.array(shape = (dimZ, dimZ))
            predictedMeasureMean : np.array(shape = (dimZ,1))
            kalmanGain : np.array(shape = (dimX, dimZ))
            
    """

    sigmaStarPoints_state = forwardFunc(sigmaPoints)
    sigmaStarPoints_measure = measureFunc(sigmaStarPoints_state)

    predictedStateMean = Ws[0] * sigmaStarPoints_state[0] + Ws[1] * np.sum(sigmaStarPoints_state[1:], axis = 0)
    predictedMeasureMean = Ws[0] * sigmaStarPoints_measure[0] + Ws[1] * np.sum(sigmaStarPoints_measure[1:], axis = 0)
    
    predictedStateMean = np.expand_dims(predictedStateMean, axis=1)
    predictedMeasureMean = np.expand_dims(predictedMeasureMean, axis=1)
    


    predictedStateCovariance = None
    Pzz = None
    Pxz = None    

    for i,sigmaStarPoint_state in enumerate(sigmaStarPoints_state):

        sigmaStarPoint_measure = np.expand_dims(sigmaStarPoints_measure[i], axis=1)
        sigmaStarPoint_state = np.expand_dims(sigmaStarPoint_state, axis=1)

        if(predictedStateCovariance is None):
            predictedStateCovariance = Wc[0] * np.dot((sigmaStarPoint_state - predictedStateMean), (sigmaStarPoint_state-predictedStateMean).T)
            Pzz = Wc[0] * np.dot((sigmaStarPoint_measure - predictedMeasureMean), (sigmaStarPoint_measure - predictedMeasureMean).T)
            Pxz = Wc[0] * np.dot((sigmaStarPoint_state - predictedStateMean), (sigmaStarPoint_measure - predictedMeasureMean).T)
        else:
            predictedStateCovariance += Wc[1] * np.dot((sigmaStarPoint_state - predictedStateMean), (sigmaStarPoint_state-predictedStateMean).T)
            Pzz += Wc[1] * np.dot((sigmaStarPoint_measure - predictedMeasureMean), (sigmaStarPoint_measure - predictedMeasureMean).T)
            Pxz += Wc[1] * np.dot((sigmaStarPoint_state - predictedStateMean), (sigmaStarPoint_measure - predictedMeasureMean).T)


    predictedStateCovariance += processNoise
    Pzz += measurementNoise
    
    kalmanGain = np.dot(Pxz, np.linalg.pinv(Pzz))

    return (predictedStateMean, predictedStateCovariance, Pzz, predictedMeasureMean, kalmanGain)





def forwardFunc(sigmaPoints):
    return sigmaPoints

def measureFunc(sigmaPoints):
    return np.array(sigmaPoints)[:,0:commonVar.dimZ]


predictedStateMean, predictedStateCovariance, Pzz, predictedMeasureMean, kalmanGain = \
    calculatePredictedState(forwardFunc, measureFunc, gSP.sigmaPoints, gUW.Ws, gUW.Wc, commonVar.processNoise, commonVar.measurementNoise)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    