from scipy.linalg import cholesky
import numpy as np

import generateSigmaPoints as gSP
import generateUnscentedWeights as gUW

print_calculatePriorState = True

def print_(*element):
    if(print_calculatePriorState):
        print(element)


def calculatePriorState(forwardFunc, measureFunc, sigmaPoints, Ws, Wc, processNoise, measurementNoise):

    """
        forwardFunc: function
        measureFunc: function
        sigmaPoints : [ np.array(shape = (dimX,1)) ]
        Ws : [Ws0, Wsi] -> only 2 elements
        Wc : [Wc0, Wci] -> only 2 elements
        processNoise : np.array(shape = (dimX, dimX))
        measurementNoise : np.array(shape = (dimZ, dimZ))
    """

    sigmaStarPoints_state = forwardFunc(sigmaPoints)
    sigmaStarPoints_measure = measureFunc(sigmaStarPoints_state)

    stateMean = Ws[0] * sigmaStarPoints_state[0] + Ws[1] * np.sum(sigmaStarPoints_state[1:], axis = 0)
    measureMean = Ws[0] * sigmaStarPoints_measure[0] + Ws[1] * np.sum(sigmaStarPoints_measure[1:], axis = 0)


    stateCovariance = None
    Pzz = None
    Pxz = None    

    for i,sigmaStarPoint_state in enumerate(sigmaStarPoints_state):

        sigmaStarPoint_measure = sigmaStarPoints_measure[i]

        if(stateCovariance is None):
            stateCovariance = Wc[0] * np.dot((sigmaStarPoint_state - stateMean), (sigmaStarPoint_state-stateMean).T)
            Pzz = Wc[0] * np.dot((sigmaStarPoint_measure - measureMean), (sigmaStarPoint_measure - measureMean).T)
            Pxz = Wc[0] * np.dot((sigmaStarPoint_state - stateMean), (sigmaStarPoint_measure - measureMean).T)
        else:
            stateCovariance += Wc[1] * np.dot((sigmaStarPoint_state - stateMean), (sigmaStarPoint_state-stateMean).T)
            Pzz += Wc[1] * np.dot((sigmaStarPoint_measure - measureMean), (sigmaStarPoint_measure - measureMean).T)
            Pxz += Wc[1] * np.dot((sigmaStarPoint_state - stateMean), (sigmaStarPoint_measure - measureMean).T)


    stateCovariance += processNoise
    Pzz += measurementNoise

    kalmanGain = np.dot(Pxz, np.linalg.pinv(Pzz))

    return (stateMean, stateCovariance, Pzz, measureMean, kalmanGain)




dimX = 5
dimZ = 3

def forwardFunc(sigmaPoints):
    return sigmaPoints

def measureFunc(sigmaPoints):
    return np.array(sigmaPoints)[:,0:dimZ]



processNoise = np.eye(dimX)
measurementNoise = np.eye(dimZ)

stateMean, stateCovariance, Pzz, measureMean, kalmanGain = \
    calculatePriorState(forwardFunc, measureFunc, gSP.sigmaPoints, gUW.Ws, gUW.Wc, processNoise, measurementNoise)