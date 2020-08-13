import numpy as np
import sys
sys.path.append("../")
import commonVariables as commonVar
import common as common

print_updateModeProbabilities_PDA = True

def print_(*element):
    if(print_updateModeProbabilities_PDA):
        print(element)
        
  

def updateModeProbabilities_PDA(modeSs, likelihoods, gateThreshold, PD, transitionMatrix, previousModeProbabilities): #checkCount : 0

    """
        Description : 
            It calculates the new mode probabilities using probability data association [BYL95 page211 4.4.1-2]

        Input:
            modeSs : np.array(shape = (Nr, dimZ, dimZ))
            likelihoods : np.array(shape = (Nr, m_k)
            gateThreshold : float between(0,1)
            PD : float between(0,1)
            transitionMatrix : np.array(shape = (Nr, Nr))
            previousModeProbabilities : np.array(shape = (Nr,1))

        Output:
            updatedModeProbabilities : np.array(shape = (Nr,1))
    """


    maximumVolume = None

    maxPzzDeterminant = 0
    for modeS in modeSs:
        det = np.linalg.det(modeS)
        if(det > maxPzzDeterminant):
            maxPzzDeterminant = det

    nz = modeSs[0].shape[0]
    maximumVolume = common.calculateNDimensionalUnitVolume(nz) * np.power(gateThreshold, nz/2) * np.sqrt(maxPzzDeterminant) #[BYL95 page 130 3.4.1-6]
    
    m_k = likelihoods.shape[1]

    summedLikelihoods = np.expand_dims(np.sum(likelihoods, axis = 1), axis=1)

    modeLambdas = (1 - PD) / np.power(maximumVolume, m_k) + PD / (m_k * np.power(maximumVolume, m_k-1)) * summedLikelihoods

    modeSwitchingProbs = np.dot(previousModeProbabilities.T, transitionMatrix).T
    
    normalizationConstant = np.dot(modeLambdas.T, modeSwitchingProbs)

    updatedModeProbabilities = np.multiply(modeSwitchingProbs, modeLambdas) / normalizationConstant

    return updatedModeProbabilities



#playground:



updatedModeProbabilities = updateModeProbabilities_PDA(commonVar.modeSs, commonVar.measurements, commonVar.likelihoods, commonVar.gateThreshold, commonVar.PD, commonVar.transitionMatrix, commonVar.previousModeProbabilities)
print_(updatedModeProbabilities)
print_(np.sum(updatedModeProbabilities))


