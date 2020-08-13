from scipy.linalg import cholesky
import numpy as np
import matplotlib.pyplot as plt

import generateUnscentedWeights as gUW

import sys
sys.path.append("../")
import commonVariables as commonVar

print_generateSigmaPoints = True

def print_(*element):
    if(print_generateSigmaPoints):
        print(element)

def generateSigmaPoints(stateMean, stateCovariance, lambda_): #checkCount : 1

    """
        Description:
            [WR00 Equation 15]
        Input:
            stateMean: np.array(shape = (dimX,1))
            stateCovariance: np.array(shape = (dimX, dimX))
            lambda_: float
            
        Output:
            sigmaPoints : np.array(shape = (dimX*2 + 1, dimX, 1))
        
    """ 

    L = stateMean.shape[0]

    sigmaPoints = [stateMean]

    sqrtMatrix = cholesky((L + lambda_) * stateCovariance)

    for i in range(L):
        sigmaPoints.append( stateMean + np.expand_dims(sqrtMatrix[i],axis=1) )
        sigmaPoints.append( stateMean - np.expand_dims(sqrtMatrix[i],axis=1) )
        
    
    return np.array(sigmaPoints, dtype = "float")




sigmaPoints = generateSigmaPoints(np.expand_dims(commonVar.stateMeans[0],axis=1), commonVar.stateCovariances[0], gUW.lambda_)
print_(sigmaPoints.shape)

plt.scatter(sigmaPoints[:,0,0], sigmaPoints[:,1,0])
