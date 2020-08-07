from scipy.linalg import cholesky
import numpy as np

print_generateSigmaPoints = False

def print_(*element):
    if(print_generateSigmaPoints):
        print(element)

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



dimX = 5
lambda_ = (0.01 **2) * (dimX + 0) - dimX

stateMean = np.random.randn(dimX,1)
stateCovariance = np.random.randn(dimX, dimX)
stateCovariance = 0.5*(stateCovariance + stateCovariance.T) / ( np.max(abs(stateCovariance))) + dimX * np.eye(dimX)

print_(stateCovariance)


sigmaPoints = generateSigmaPoints(stateMean, stateCovariance, lambda_)
print_(sigmaPoints.shape)
