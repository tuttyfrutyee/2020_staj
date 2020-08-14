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


def calculatePredictedState(forwardFunc, dt, measureFunc, sigmaPoints, Ws, Wc, processNoise, measurementNoise):

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
            S : np.array(shape = (dimZ, dimZ))
            predictedMeasureMean : np.array(shape = (dimZ,1))
            kalmanGain : np.array(shape = (dimX, dimZ))
            
    """

    sigmaStarPoints_state = []
    sigmaStarPoints_measure = []
    
    for sigmaPoint in sigmaPoints:
        
        sigmaStarPoint_state = forwardFunc(sigmaPoint,dt)
        sigmaStarPoint_measure = measureFunc(sigmaStarPoint_state)

        sigmaStarPoints_state.append(sigmaStarPoint_state)
        sigmaStarPoints_measure.append(sigmaStarPoint_measure)
        
    sigmaStarPoints_state = np.array(sigmaStarPoints_state)
    sigmaStarPoints_measure = np.array(sigmaStarPoints_measure)        
        
    predictedStateMean = Ws[0] * sigmaStarPoints_state[0] + Ws[1] * np.sum(sigmaStarPoints_state[1:], axis = 0)
    predictedMeasureMean = Ws[0] * sigmaStarPoints_measure[0] + Ws[1] * np.sum(sigmaStarPoints_measure[1:], axis = 0)
    
    


    predictedStateCovariance = None
    Pzz = None
    Pxz = None    

    for i,sigmaStarPoint_state in enumerate(sigmaStarPoints_state):

        sigmaStarPoint_measure = sigmaStarPoints_measure[i]

        if(predictedStateCovariance is None):
            predictedStateCovariance = Wc[0] * np.dot((sigmaStarPoint_state - predictedStateMean), (sigmaStarPoint_state-predictedStateMean).T)
            Pzz = Wc[0] * np.dot((sigmaStarPoint_measure - predictedMeasureMean), (sigmaStarPoint_measure - predictedMeasureMean).T)
            Pxz = Wc[0] * np.dot((sigmaStarPoint_state - predictedStateMean), (sigmaStarPoint_measure - predictedMeasureMean).T)
        else:
            predictedStateCovariance += Wc[1] * np.dot((sigmaStarPoint_state - predictedStateMean), (sigmaStarPoint_state-predictedStateMean).T)
            Pzz += Wc[1] * np.dot((sigmaStarPoint_measure - predictedMeasureMean), (sigmaStarPoint_measure - predictedMeasureMean).T)
            Pxz += Wc[1] * np.dot((sigmaStarPoint_state - predictedStateMean), (sigmaStarPoint_measure - predictedMeasureMean).T)


    predictedStateCovariance += processNoise
    S = Pzz + measurementNoise
    
    kalmanGain = np.dot(Pxz, np.linalg.pinv(S))

    return (predictedStateMean, predictedStateCovariance, S, predictedMeasureMean, kalmanGain)





def forwardFunc(sigmaPoints, dt):
    return sigmaPoints

def measureFunc(sigmaPoints):
    return np.array(sigmaPoints)[0:commonVar.dimZ]


predictedStateMean, predictedStateCovariance, Pzz, predictedMeasureMean, kalmanGain = \
    calculatePredictedState(forwardFunc, commonVar.dt, measureFunc, gSP.sigmaPoints, gUW.Ws, gUW.Wc, commonVar.processNoise, commonVar.measurementNoise)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    