
import numpy as np

print_pdaPass = True

def print_(*element):
    if(print_pdaPass):
        print(element)


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




dimZ = 3
dimX = 5
m_k = 10

kalmanGain = np.random.rand(dimX, dimZ)

associationProbs = abs(np.random.randn(m_k,))
associationProbs = associationProbs / np.sum(associationProbs) - 0.3

measurements = np.random.rand(dimZ, 1)
priorStateMean = np.random.randn(dimX, 1)
priorStateMeasuredMean = np.random.randn(dimX, 1)
priorStateCovariance = np.random.randn(dimX, dimX)
Pzz = np.random.randn(dimZ, dimZ)

x_final, P_final = pdaPass(kalmanGain, associationProbs, measurements, priorStateMean, priorStateMeasuredMean, priorStateCovariance, Pzz)