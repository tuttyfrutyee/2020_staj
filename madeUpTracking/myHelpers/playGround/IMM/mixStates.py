import numpy as np
import sys
sys.path.append("../")
import commonVariables as commonVar

print_mixStates = True

def print_(*element):
    if(print_mixStates):
        print(element)


def mixStates(stateMeans, stateCovariances, transitionMatrix, modeProbs):

    """
        Description:
            Mixes(interaction) the states of different models with weights of modeProbs
            [DIMM Equation 5-6]

        Input: 
            stateMeans : np.array(shape = (Nr, dimX))
            stateCovariances : np.array(shape = (Nr, dimX, dimX))
            transitionMatrix : np.array(shape = (Nr, Nr))
            modeProbs : np.array(shape = (Nr,1))

        Output:
            mixedMeans : np.array(shape = (Nr, dimX))
            mixedCovariances : np.array(shape = (Nr, dimX, dimX))
    """

    mixingRatios = (transitionMatrix * modeProbs) / np.dot(modeProbs.T, transitionMatrix)
    print_(np.sum(mixingRatios,axis=0))
    mixedMeans = np.dot(stateMeans.T, mixingRatios).T
    mixedCovariances = []


    for i in range(transitionMatrix.shape[0]):
        
        mixedCovariance = None
        
        for j in range(transitionMatrix.shape[1]):

            if(mixedCovariance is None):
                difX = stateMeans[j] - mixedMeans[i]
                mixedCovariance = mixingRatios[j][i] * (stateCovariances[j] + np.dot(difX, difX.T))
            else:
                difX = stateMeans[j] - mixedMeans[i]
                mixedCovariance += mixingRatios[j][i] * (stateCovariances[j] + np.dot(difX, difX.T))            

        
        mixedCovariances.append(mixedCovariance)
    
    mixedCovariances = np.array(mixedCovariances, dtype="float")

    return (mixedMeans, mixedCovariances)




mixedMeans, mixedCovariances = mixStates(commonVar.stateMeans, commonVar.stateCovariances, commonVar.transitionMatrix, commonVar.modeProbs)

# playground
mixingRatios = (commonVar.transitionMatrix * commonVar.modeProbs) / np.dot(commonVar.modeProbs.T, commonVar.transitionMatrix)
