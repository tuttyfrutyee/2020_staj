
import numpy as np

import sys
sys.path.append("../")
import commonVariables as commonVar
sys.path.append("../JPDA")
import calculateMarginalAssociationProbabilities as cMAP
import createValidationMatrix as cVM

print_pdaPass = True

def print_(*element):
    if(print_pdaPass):
        print(element)


def pdaPass(kalmanGain, associationProbs, measurements, priorStateMean, priorStateMeasuredMean, priorStateCovariance, Pzz): #checkCount : 1

    """
        Description:
            It is the update function for one track with multiple measurements using probability data association
            [page 132, 3.4.2-8 to 3.4.2-12, BYL95]
        Input:
            kalmanGain : np.array( shape = (dimX, dimZ) )
            associationProbs : np.array(shape = ([B_1t, B_2t, ..., B_m_kt]
            measurements : np.array(shape = (m_k, dimZ) )
            priorStateMean : np.array(shape = (dimX, 1))
            priorStateMeasuredMean : np.array(shape = (dimZ, 1) )
            priorStateCovariance : np.array( shape = (dimX, dimX) )
            Pzz : np.array(shape = (dimZ, dimZ) )
        
        Output:
            x_updated : np.array(shape = (dimX,1))
            P_updated : np.array(shape = (dimX, dimX))
    """

    vk = []
    v_fused = None

    B0 = 1 - np.sum(associationProbs)

    P_c = priorStateCovariance - np.dot(kalmanGain, np.dot(Pzz, kalmanGain.T))
    P_tilda = None
    runningCovariance = None

    for i,measurement in enumerate(measurements):
        measurement = np.expand_dims(measurement, axis=1)
        vk.append(measurement - priorStateMeasuredMean)


        if(v_fused is None):
            v_fused = associationProbs[i] * vk[i]
            runningCovariance = associationProbs[i] * (np.dot(vk[i], vk[i].T))
        else:
            v_fused += associationProbs[i] * vk[i]
            runningCovariance += associationProbs[i] * (np.dot(vk[i], vk[i].T))


    P_tilda = np.dot(kalmanGain, np.dot( runningCovariance - np.dot(v_fused, v_fused.T) , kalmanGain.T))
    print_("v_fused shape : ",v_fused.shape)
    x_updated = priorStateMean + np.dot(kalmanGain, v_fused)
    P_updated = B0 * priorStateCovariance + [1-B0] * P_c  + P_tilda

    return (x_updated, P_updated)


x_updated, P_updated = pdaPass(commonVar.kalmanGain, cMAP.marginalAssociationProbabilities[:,0], cVM.validatedMeasurements, commonVar.priorStateMean, commonVar.priorStateMeasuredMean, commonVar.priorStateCovariance, commonVar.Pzz)
print_("x_updated shape: " ,x_updated.shape)
print_("P_updated shape: " ,P_updated.shape)