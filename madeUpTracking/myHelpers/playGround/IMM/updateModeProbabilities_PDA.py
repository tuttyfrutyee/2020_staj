import numpy as np
import sys
sys.path.append("../")
import commonVariables as common

print_updateModeProbabilities_PDA = True

def print_(*element):
    if(print_updateModeProbabilities_PDA):
        print(element)
        
def calculateNDimensionalUnitVolume(n): #checkCount : 1, confident
    return np.power(np.pi, n/2) / gamaFunc(n/2 + 1)

def gamaFunc(n): #checkCount : 1, confident

    gama_1over2 = np.sqrt(np.pi)
    runningProduct = 1
    
    while(n > 1):
        runningProduct *= (n-1)
        n = n - 1
    if(n == 0.5):
        runningProduct *= gama_1over2
    return runningProduct        

def updateModeProbabilities_PDA(modePzzs, measurements, likelihoods, gateThreshold, PD, transitionMatrix, previousModeProbabilities): #checkCount : 0

    """
        Description : 
            It calculates the new mode probabilities using probability data association [BYL95 page211 4.4.1-2]

        Input:
        modePzzs : np.array(shape = (Nr, dimZ, dimZ))
        measurements : np.array(shape = (m_k, dimZ))
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
    for modePzz in modePzzs:
        det = np.linalg.det(modePzz)
        if(det > maxPzzDeterminant):
            maxPzzDeterminant = det

    nz = modePzzs[0].shape[0]
    maximumVolume = calculateNDimensionalUnitVolume(nz) * np.power(gateThreshold, nz/2) * np.sqrt(maxPzzDeterminant) #[BYL95 page 130 3.4.1-6]
    
    m_k = measurements.shape[0]

    summedLikelihoods = np.expand_dims(np.sum(likelihoods, axis = 1), axis=1)

    modeLambdas = (1 - PD) / np.power(maximumVolume, m_k) + PD / (m_k * np.power(maximumVolume, m_k-1)) * summedLikelihoods

    modeSwitchingProbs = np.dot(previousModeProbabilities.T, transitionMatrix).T
    
    normalizationConstant = np.dot(modeLambdas.T, modeSwitchingProbs)

    updatedModeProbabilities = np.multiply(modeSwitchingProbs, modeLambdas) / normalizationConstant

    return updatedModeProbabilities



#playground:

print_(calculateNDimensionalUnitVolume(1), "should be = ",2)
print_(calculateNDimensionalUnitVolume(2), "should be = ",np.pi)
print_(calculateNDimensionalUnitVolume(3), "should be = ", np.pi * 4 / 3)

updatedModeProbabilities = updateModeProbabilities_PDA(common.modePzzs, common.measurements, common.likelihoods, common.gateThreshold, common.PD, common.transitionMatrix, common.previousModeProbabilities)
print_(updatedModeProbabilities)
print_(np.sum(updatedModeProbabilities))


