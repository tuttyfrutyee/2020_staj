# -*- coding: utf-8 -*-
import numpy as np

import sys
sys.path.append("../")
import commonVariables as commonVar

print_fuseModelStates = True

def print_(*element):
    if(print_fuseModelStates):
        print(element)

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



fusedStateMean, fusedStateCovariance = fuseModelStates(commonVar.stateMeans, commonVar.stateCovariances, commonVar.modeProbs)