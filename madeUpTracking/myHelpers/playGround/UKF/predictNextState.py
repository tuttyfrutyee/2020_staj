# -*- coding: utf-8 -*-

import numpy as np

def predictNextState(forwardFunc, dt, sigmaPoints, Ws, Wc, processNoise):

    """
        Description:
            Predicts the next state
        
        Input:
            forwardFunc : function
            dt: float
            sigmaPoints : np.array(shape = (2*dimX +1, dimX, 1))
            Ws : [Ws0, Wsi] -> only 2 elements
            Wc : [Wc0, Wci] -> only 2 elements
            processNoise : np.array(shape = (dimX, dimX))
        
        Output:
            predictedStateMean : np.array(shape = (dimX, 1))
            predictedStateCovariance : np.array(shape = (dimX, dimX))
    """
    
    sigmaStarPoints_state = []
    
    for sigmaPoint in sigmaPoints:
        sigmaStarPoints_state.append(forwardFunc(sigmaPoint,dt))

    
    sigmaStarPoints_state = np.array(sigmaStarPoints_state)

    predictedStateMean = Ws[0] * sigmaStarPoints_state[0] + Ws[1] * np.sum(sigmaStarPoints_state[1:], axis = 0)



    predictedStateCovariance = None 

    for i,sigmaStarPoint_state in enumerate(sigmaStarPoints_state):
        
        if(predictedStateCovariance is None):
            predictedStateCovariance = Wc[0] * np.dot((sigmaStarPoint_state - predictedStateMean), (sigmaStarPoint_state-predictedStateMean).T)
        else:
            predictedStateCovariance += Wc[1] * np.dot((sigmaStarPoint_state - predictedStateMean), (sigmaStarPoint_state-predictedStateMean).T)
    
    
    predictedStateCovariance += processNoise  

    return (predictedStateMean, predictedStateCovariance)  