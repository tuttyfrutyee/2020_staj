import numpy as np
import myHelpers.common as common
from scipy.stats import norm, multivariate_normal

"""
    References : 

        "DIMM : Derivation of the IMM filter" inside related papers folder

"""

##############################################################################
#Common
##############################################################################

def mixStates(stateMeans, stateCovariances, transitionMatrix, modeProbs): #checkCount : 1

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


def fuseModelStates(stateMeans, stateCovariances, modeProbabilities): #checkCount : 1

    """
        Description:
            It fuses the states of the multiple models. Note that this is only for output purposes.
            Internally the seperate states of the models should be kept, and not fused.
            [DIMM Equation 53]
        
        Input:
            stateMeans : np.array(shape = (Nr, dimX))
            stateCovariances : np.array(shape = (Nr, dimX, dimX))
            modeProbabilities : np.array(shape = (Nr,1))

        Output:
            fusedStateMean : np.array(shape = (dimX, 1))
            fusedStateCovariance : np.array(shape = (dimX, dimX))
    """

    fusedStateMean = np.dot(stateMeans.T, modeProbabilities)

    fusedStateCovariance = None

    for i,(stateMean,stateCovariance) in enumerate(zip(stateMeans, stateCovariances)):

        stateMean = np.expand_dims(stateMean, axis=1)
        
        if(fusedStateCovariance is None):
            fusedStateCovariance = modeProbabilities[i] * (stateCovariance + np.dot(stateMean - fusedStateMean, (stateMean - fusedStateMean).T) )
        else:
            fusedStateCovariance += modeProbabilities[i] * (stateCovariance + np.dot(stateMean - fusedStateMean, (stateMean - fusedStateMean).T) )

    return (fusedStateMean, fusedStateCovariance)    

##############################################################################
# With Data Association
##############################################################################

def updateModeProbabilities_PDA(modeStateMeans_measured, modeSs, measurements, gateThreshold, PD, transitionMatrix, previousModeProbabilities): #checkCount : 1

    """
        Description : 
            It calculates the new mode probabilities using probability data association [BYL95 page211 4.4.1-2]

        Input:
            modeStateMeans_measured : np.array(shape = (Nr, dimZ))
            modeSs : np.array(shape = (Nr, dimZ, dimZ))
            measurements : np.array(shape = (m_k, dimZ))
            gateThreshold : float between(0,1)
            PD : float between(0,1)
            transitionMatrix : np.array(shape = (Nr, Nr))
            previousModeProbabilities : np.array(shape = (Nr,1))

        Output:
            updatedModeProbabilities : np.array(shape = (Nr,1))
    """

    Nr = modeSs.shape[0]
    m_k = measurements.shape[0]

    likelihoods = np.zeros((Nr, m_k))

    for i, modeStateMean_measured in enumerate(modeStateMeans_measured):
        for j, measurement in enumerate(measurements):
            
            modeS = modeSs[i]

            likelihoods[i][j] =  multivariate_normal.pdf(measurement.flatten(), modeStateMean_measured.flatten(), modeS, True)     

    maximumVolume = None

    maxPzzDeterminant = 0
    for modeS in modeSs:
        det = np.linalg.det(modeS)
        assert(det > 0)
        if(det > maxPzzDeterminant):
            maxPzzDeterminant = det

    nz = modeSs[0].shape[0]
    maximumVolume = common.calculateNDimensionalUnitVolume(nz) * np.power(gateThreshold, nz/2) * np.sqrt(maxPzzDeterminant) #[BYL95 page 130 3.4.1-6]
    
    summedLikelihoods = np.expand_dims(np.sum(likelihoods, axis = 1), axis=1)

    modeLambdas = (1 - PD) / np.power(maximumVolume, m_k) + PD / (m_k * np.power(maximumVolume, m_k-1)) * summedLikelihoods

    modeSwitchingProbs = np.dot(previousModeProbabilities.T, transitionMatrix).T
    
    normalizationConstant = np.dot(modeLambdas.T, modeSwitchingProbs)

    updatedModeProbabilities = np.multiply(modeSwitchingProbs, modeLambdas) / normalizationConstant

    return updatedModeProbabilities


    

##############################################################################
# With Single Measurement
##############################################################################

def updateModeProbabilities(modeStateMeans_measured, modeSs, measurement, transitionMatrix, previousModeProbs): #checkCount : 1
    
    """
        Description : 
            It calculates the new mode probabilities, considering single measurement
        Input:
            modeStateMeans_measured: np.array(shape = (Nr, dimZ))
            modeSs : np.array(shape = (Nr, dimZ, dimZ))
            measurement : np.array(shape = (dimZ, 1))
            transitionMatrix : np.array(shape = (Nr, Nr))
            previousModeProbs : np.array(shape = (Nr,1))

        Output:
            updatedModeProbabilities : np.array(shape = (Nr,1))        

    """


    Nr = modeStateMeans_measured.shape[0]

    likelihoods = np.zeros((Nr,1))

    for i,(modeStateMean_measured, modeS) in enumerate(zip(modeStateMeans_measured, modeSs)):    

        likelihoods[i] = multivariate_normal.pdf(measurement.flatten(), modeStateMean_measured.flatten(), modeS, True)         

    modeSwitch_plainMarkovProbs = np.dot(previousModeProbs.T, transitionMatrix).T
    

    updatedModeProbabilities = modeSwitch_plainMarkovProbs * likelihoods

    #normalize
    updatedModeProbabilities = updatedModeProbabilities / np.sum(updatedModeProbabilities)

    return updatedModeProbabilities












