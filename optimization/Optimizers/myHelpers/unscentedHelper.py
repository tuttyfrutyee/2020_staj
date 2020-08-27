import torch
import numpy as np


"""
    References: 
        [WR00] The Unscented Kalman Filter for Nonlinear Estimation, Eric A. Wan and Rudolph van der Merwe
"""

def generateUnscentedWeights(L, alpha, beta, kappa): #checkCount : 1
    """
        Description:
            [WR00 Equation 15]
        Input:
            L: float
            alpha: float
            beta: float
            kappa: float
    """

    lambda_ = (alpha ** 2) * (L + kappa) - L


    Ws0 = lambda_ / (L + lambda_)
    Wc0 = lambda_ / (L + lambda_) + (1 - alpha**2 + beta)

    Wsi = 0.5 / (L + lambda_)
    Wci = 0.5 / (L + lambda_)

    return ([Ws0, Wsi],[Wc0, Wci], lambda_)

def generateSigmaPoints(stateMean, stateCovariance, lambda_): #checkCount : 1

    """
        Description:
            [WR00 Equation 15]
        Input:
            stateMean: torch.tensor(shape = (dimX,1))
            stateCovariance: torch.tensor(shape = (dimX, dimX))
            lambda_: float
        
        Output:
            sigmaPoints : torch.tensor(shape = (dimX*2 + 1, dimX, 1))

    """

    L = stateMean.shape[0]

    sigmaPoints = [stateMean.unsqueeze(0)]

    sqrtMatrix = torch.cholesky((L + lambda_) * stateCovariance)

    for i in range(L):
        sigmaPoints.append( (stateMean + sqrtMatrix[i].unsqueeze(1)).unsqueeze(0) )
        sigmaPoints.append( (stateMean - sqrtMatrix[i].unsqueeze(1)).unsqueeze(0) )


    return torch.cat(sigmaPoints, dim=0)

def predictNextState(forwardFunc, dt, sigmaPoints, Ws, Wc, processNoise):

    """
        Description:
            Predicts the next state
        
        Input:
            forwardFunc : function
            dt: float
            sigmaPoints : torch.tensor(shape = (2*dimX +1, dimX, 1))
            Ws : [Ws0, Wsi] -> only 2 elements
            Wc : [Wc0, Wci] -> only 2 elements
            processNoise : torch.tensor(shape = (dimX, dimX))
        
        Output:
            predictedStateMean : torch.tensor(shape = (dimX, 1))
            predictedStateCovariance : torch.tensor(shape = (dimX, dimX))
    """
    
    sigmaStarPoints_state = []
    
    for sigmaPoint in sigmaPoints:
        sigmaStarPoints_state.append(forwardFunc(sigmaPoint,dt).unsqueeze(0))

    
    sigmaStarPoints_state = torch.cat(sigmaStarPoints_state, dim=0)
    
    
    predictedStateMean = Ws[0] * sigmaStarPoints_state[0] + Ws[1] * torch.sum(sigmaStarPoints_state[1:], dim = 0)



    predictedStateCovariance = None 

    for i,sigmaStarPoint_state in enumerate(sigmaStarPoints_state):
        
        if(predictedStateCovariance is None):
            predictedStateCovariance = Wc[0] * torch.mm((sigmaStarPoint_state - predictedStateMean), (sigmaStarPoint_state-predictedStateMean).T)
        else:
            predictedStateCovariance += Wc[1] * torch.mm((sigmaStarPoint_state - predictedStateMean), (sigmaStarPoint_state-predictedStateMean).T)
    
    
    predictedStateCovariance += processNoise  

    return (predictedStateMean, predictedStateCovariance)  

    

def calculateUpdateParameters(predictedStateMean, predictedStateCovariance, measureFunc, sigmaPoints, Ws, Wc, measurementNoise): #checkCount : 1

    """
    
        Description : 
            Calculates the predicted state
            [WR00 Algorithm 3.1]
    
        Input:

            predictedStateMean : torch.tensor(shape = (dimX, 1))
            predictedStateCovariance : torch.tensor(shape = (dimX, dimX))
            measureFunc : function
            sigmaPoints : torch.tensor(shape = (2*dimX + 1, dimX, 1))
            Ws : [Ws0, Wsi] -> only 2 elements
            Wc : [Wc0, Wci] -> only 2 elements
            measurementNoise : torch.tensor(shape = (dimZ, dimZ))
        
        Output:
            S : torch.tensor(shape = (dimZ, dimZ))
            kalmanGain : torch.tensor(shape = (dimX, dimZ))
            predictedMeasureMean : torch.tensor(shape = (dimZ,1))

            
    """
    sigmaStarPoints_measure = []
    
    for sigmaPoint in sigmaPoints:
        
        sigmaStarPoint_measure = measureFunc(sigmaPoint)

        sigmaStarPoints_measure.append(sigmaStarPoint_measure.unsqueeze(0))
    
    sigmaStarPoints_measure = torch.cat(sigmaStarPoints_measure, dim=0)

    predictedMeasureMean = Ws[0] * sigmaStarPoints_measure[0] + Ws[1] * torch.sum(sigmaStarPoints_measure[1:], dim = 0)
    

    Pzz = None
    Pxz = None

    for i, sigmaPoint in enumerate(sigmaPoints):

        sigmaStarPoint_measure = sigmaStarPoints_measure[i]

        if(Pzz is None):
            Pzz = Wc[0] * torch.mm((sigmaStarPoint_measure - predictedMeasureMean), (sigmaStarPoint_measure - predictedMeasureMean).T)
            Pxz = Wc[0] * torch.mm((sigmaPoint - predictedStateMean), (sigmaStarPoint_measure - predictedMeasureMean).T)
        else:
            Pzz += Wc[1] * torch.mm((sigmaStarPoint_measure - predictedMeasureMean), (sigmaStarPoint_measure - predictedMeasureMean).T)
            Pxz += Wc[1] * torch.mm((sigmaPoint - predictedStateMean), (sigmaStarPoint_measure - predictedMeasureMean).T)               
    
    
    S = Pzz + measurementNoise
    
    kalmanGain = torch.mm(Pxz, torch.inverse(S))

    return (S, kalmanGain, predictedMeasureMean)



