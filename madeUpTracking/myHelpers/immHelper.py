import numpy as np

"""
    References : 

        "Derivation of the IMM filter" inside related papers folder

"""

def mixStates(stateMeans, stateCovariances, transitionMatrix, modeProbs):

    """
        stateMeans : np.array(shape = (Nr, dimX))
        stateCovariances : np.array(shape = (Nr, dimX, dimX))
        transitionMatrix : np.array(shape = (Nr, Nr))
        modeProbs : np.array(shape = (Nr,))
    """

    mixingRatios = (transitionMatrix.T * modeProbs).T / np.dot(modeProbs, transitionMatrix) #not sure about this one
    mixedMeans = []
    mixedCovariances = []


    for i in range(transitionMatrix.shape[0]):
        
        mixedMean = None
        mixedCovariance = None

        for j in range(transitionMatrix.shape[1]):

            if(mixedMean is None):
                mixedMean = mixingRatios[j][i] * stateMeans[j]
                difX = stateMeans[j] - mixedMean
                mixedCovariance = mixingRatios[j][i] * (stateCovariances[j] + np.dot(difX, difX.T))
            else:
                mixedMean += mixingRatios[j][i] * stateMeans[j]
                difX = stateMeans[j] - mixedMean
                mixedCovariance += mixingRatios[j][i] * (stateCovariances[j] + np.dot(difX, difX.T))            

        
        mixedMeans.append(mixedMean)
        mixedCovariances.append(mixedCovariance)

    return (mixedMeans, mixedCovariances)