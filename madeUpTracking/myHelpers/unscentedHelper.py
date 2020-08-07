from scipy.linalg import cholesky
import numpy as np


"""
    References: 
        The Unscented Kalman Filter for Nonlinear Estimation, Eric A. Wan and Rudolph van der Merwe
"""

def generateUnscentedWeights(L, alpha, beta, kappa):

    lambda_ = (alpha ** 2) * (L + kappa) - L


    Ws0 = lambda_ / (L + lambda_)
    Wc0 = lambda_ / (L + lambda_) + (1 - alpha**2 + beta)

    Wsi = 0.5 / (L + lambda_)
    Wci = 0.5 / (L + lambda_)

    return ([Ws0, Wsi],[Wc0, Wci], lambda_)

def generateSigmaPoints(stateMean, stateCovariance, lambda_):

    """
        stateMean : np.array(shape = (dimX,1))
        stateCovariance : np.array(shape = (dimX, dimX))
        lambda_ : float
    """

    L = stateMean.shape[0]

    sigmaPoints = [stateMean]

    sqrtMatrix = cholesky((L + lambda_) * stateCovariance)

    for i in range(L):
        sigmaPoints.append( stateMean + np.expand_dims(sqrtMatrix[i],axis=1) )
        sigmaPoints.append( stateMean - np.expand_dims(sqrtMatrix[i],axis=1) )
        

    return np.array(sigmaPoints, dtype="float")


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
    
    kalmanGain = np.dot(Pxz, np.linalg.pinv(Pzz ))

    return (stateMean, stateCovariance, Pzz, measureMean, kalmanGain)



