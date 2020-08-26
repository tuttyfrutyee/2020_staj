from scipy.linalg import cholesky
import numpy as np
import matplotlib.pyplot as plt

import generateUnscentedWeights as gUW
import predictNextState as pNS

import sys
sys.path.append("../")
import commonVariables as commonVar

print_generateSigmaPoints = True

def print_(*element):
    if(print_generateSigmaPoints):
        print(element)

def generateSigmaPoints(stateMean, stateCovariance, lambda_): #checkCount : 1

    """
        Description:
            [WR00 Equation 15]
        Input:
            stateMean: np.array(shape = (dimX,1))
            stateCovariance: np.array(shape = (dimX, dimX))
            lambda_: float
            
        Output:
            sigmaPoints : np.array(shape = (dimX*2 + 1, dimX, 1))
        
    """ 

    L = stateMean.shape[0]

    sigmaPoints = [stateMean]

    sqrtMatrix = cholesky((L + lambda_) * stateCovariance)

    for i in range(L):
        sigmaPoints.append( stateMean + np.expand_dims(sqrtMatrix[i],axis=1) )
        sigmaPoints.append( stateMean - np.expand_dims(sqrtMatrix[i],axis=1) )
        
    
    return np.array(sigmaPoints, dtype = "float")


#playground

#sigmaPoints = generateSigmaPoints(np.expand_dims(commonVar.stateMeans[0],axis=1), commonVar.stateCovariances[0], gUW.lambda_)
#print_(sigmaPoints.shape)

#plt.scatter(sigmaPoints[:,0,0], sigmaPoints[:,1,0])

def massageToCovariance(P, scale):
    return 1/2*(P + P.T) + np.eye(P.shape[0]) * scale

def putAngleInRange(angle):
    
    angle = angle % (2*np.pi)
    
    if(angle > (np.pi)):
        angle -= 2*np.pi
    elif(angle < (-np.pi)):
        angle += 2*np.pi
        
    return angle


def f_predict_model2(x_, dt):

    
    x = np.copy(x_)
    
    x[2] = putAngleInRange(x[2])
    x[4] = putAngleInRange(x[4])
    
    X_new = np.copy(x)
            
    x_new = x[0] + x[3] / x[4] * ( -np.sin(x[2]) + np.sin( x[2] + dt * x[4] ) )
    y_new = x[1] + x[3] / x[4] * ( np.cos(x[2])  - np.cos( x[2] + dt * x[4] ) )
    
    phi_new = x[2] + dt * x[4] 
    

    phi_new = putAngleInRange(phi_new)
    
    X_new[0] = x_new
    X_new[1] = y_new
    X_new[2] = phi_new
    
    return X_new

Ws, Wc, lambda_ = gUW.generateUnscentedWeights(L = 5, alpha = 2.5e-3, beta = 2, kappa = 0)

P1 = np.array([[ 1.39840606,  0.34257742,  1.03632606,  0.39334248,  1.54489528],
       [ 0.34257742,  1.17497714,  0.16570133,  0.08224784, -0.23709047],
       [ 1.03632606,  0.16570133,  1.23428963,  0.32595831,  2.7590724 ],
       [ 0.39334248,  0.08224784,  0.32595831,  3.03269858, -0.61742364],
       [ 1.54489528, -0.23709047,  2.7590724 , -0.61742364, 14.68639834]])

x1 = np.array([[42.99978159],
       [35.56088666],
       [-3.42200573],
       [ 5.77163446],
       [-3.30217425]])



P2 = np.array([[ 1.58234790e+06,  1.49518048e+07,  4.82870900e+08,
         7.02420406e-01, -5.36523206e+08],
       [ 1.49518048e+07,  1.41281593e+08,  4.56271129e+09,
         5.97617693e+00, -5.06967908e+09],
       [ 4.82870900e+08,  4.56271129e+09,  1.47353481e+11,
         1.93372500e+02, -1.63726085e+11],
       [ 7.02420406e-01,  5.97617693e+00,  1.93372500e+02,
         3.03343079e+00, -2.15182184e+02],
       [-5.36523206e+08, -5.06967908e+09, -1.63726085e+11,
        -2.15182184e+02,  1.81917868e+11]])

x2 = np.array([[-8.46940150e+02],
       [-8.36916083e+03],
       [-2.71430879e+05],
       [ 5.65169610e+00],
       [ 3.01589760e+05]])


P1 = massageToCovariance(P1, 1e-8)

sigmaPoints = generateSigmaPoints(x1, P1, lambda_)
print_(sigmaPoints.shape)

dt = 0.1

processNoise =     np.array([
        
       [0.114736907423371,0,0,0,0],
       [0,0.1354455356615292,0,0,0],
       [0,0,0.6637200640035631,0,0],
       [0,0,0,2.9248106675773875,0],
       [0,0,0,0,0.9305139758546961]      
     
        # [Q_0, 0, 0, 0, 0],
        # [0, Q_0, 0, 0, 0],
        # [0, 0, Q_0 / 1e2, 0, 0],
        # [0, 0, 0, Q_0, 0],
        # [0, 0, 0, 0, Q_0 / 1e8] 
     
       # [0,0,0,0,0],
       # [0,0,0,0,0],
       # [0,0,0,0,0],
       # [0,0,0,0,0],
       # [0,0,0,0,0],     
         
     ])/ 4000

predictedStateMean, predictedStateCovariance = pNS.predictNextState(f_predict_model2, dt, sigmaPoints, Ws, Wc, processNoise)
print(predictedStateMean)
print(predictedStateCovariance)