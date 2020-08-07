
import numpy as np

"""
    References:

    [BYL95] : MultiTarget-Multisensor Tracking Yaakov Bar-Shalom 1995

"""

def pdaPass(kalmanGain, associationProbs, measurements, priorStateMean, priorStateMeasuredMean, priorStateCovariance, Pzz):

    """
        kalmanGain : np.array( shape = (dim(x), dim(z)) )
        associationProbs : [B_1t, B_2t, ..., B_m(k)t]
        measurements : np.array(shape = (m(k), dim(z), 1) )
        priorStateMean : np.array(shape = (dim(x), 1))
        priorStateMeasuredMean : np.array(shape = (dim(z), 1) )
        priorStateCovariance : np.array( shape = (dim(x), dim(x)) )

    """

    vk = []
    v_fused = None

    B0 = 1 - np.sum(associationProbs)

    P_c = priorStateCovariance - np.dot(kalmanGain, np.dot(Pzz, kalmanGain.T))
    P_tilda = None
    runningCovariance = None
    
    P_final = None
    x_final = None  

    for i,measurement in enumerate(measurements):
        vk.append(measurement - priorStateMeasuredMean)

        if(v_fused is None):
            v_fused = associationProbs[i] * vk[i]
            runningCovariance = associationProbs[i] * (np.dot(vk[i], vk[i].T))
        else:
            v_fused += associationProbs[i] * vk[i]
            runningCovariance += associationProbs[i] * (np.dot(vk[i], vk[i].T))


    print_("runningCovariance.shape = ", runningCovariance.shape)
    print_("v_fused.shape = ", v_fused.shape)
    print_("v_fused dot v_fused.T shape = ", np.dot(v_fused, v_fused.T).shape)

    P_tilda = np.dot(kalmanGain, np.dot( (runningCovariance - np.dot(v_fused, v_fused.T)) , kalmanGain.T))

    x_final = priorStateMean + np.dot(kalmanGain, v_fused)
    P_final = B0 * priorStateCovariance + [1-B0] * P_c  + P_tilda

    return (x_final, P_final)



