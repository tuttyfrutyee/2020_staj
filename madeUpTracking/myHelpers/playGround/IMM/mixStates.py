import numpy as np

print_mixStates = True

def print_(*element):
    if(print_mixStates):
        print(element)


def mixStates(stateMeans, stateCovariances, transitionMatrix, modeProbs):

    """
        stateMeans : np.array(shape = (Nr, dimX))
        stateCovariances : np.array(shape = (Nr, dimX, dimX))
        transitionMatrix : np.array(shape = (Nr, Nr))
        modeProbs : np.array(shape = (Nr,))
    """

    mixingRatios = (transitionMatrix.T * modeProbs).T / np.dot(modeProbs, transitionMatrix) #not sure about this one
    print_(np.sum(mixingRatios,axis=0))
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


dimX = 5
Nr = 3 #number of modes(models)
stateMeans = np.random.randn(Nr, dimX)
stateCovariances = np.random.randn(Nr, dimX, dimX)
transitionMatrix = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])
modeProbs = np.array([0.1, 0.5, 0.4])

mixedMeans, mixedCovariances = mixStates(stateMeans, stateCovariances, transitionMatrix, modeProbs)