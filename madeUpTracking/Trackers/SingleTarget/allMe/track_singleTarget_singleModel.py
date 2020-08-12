# -*- coding: utf-8 -*-

import numpy as np
import myHelpers.unscentedHelper as uH


measurementNoiseStd = 2
Q_0 = 0.005

#helpers

def putAngleInRange(angle):
    
    angle = angle % (2*np.pi)
    
    if(angle > (np.pi)):
        angle -= 2*np.pi
    elif(angle < (-np.pi)):
        angle += 2*np.pi
        
    return angle

def massageToCovariance(P, scale):
    return 1/2*(P + P.T) + np.eye(P.shape[0]) * scale

########


MeasurementNoiseCovs = [
    #modeltype 0
    [
        [measurementNoiseStd * measurementNoiseStd, 0],
        [0, measurementNoiseStd * measurementNoiseStd]
    ],
    #modeltype 1
    [
        [measurementNoiseStd * measurementNoiseStd, 0],
        [0, measurementNoiseStd * measurementNoiseStd]
    ],
    #modeltype 2
    [
        [measurementNoiseStd * measurementNoiseStd, 0],
        [0, measurementNoiseStd * measurementNoiseStd]
    ],
]
ProcessNoiseCovs = [
    #modeltype 0
    [
        [Q_0, 0, 0, 0],
        [0, Q_0, 0, 0],
        [0, 0, Q_0, 0],
        [0, 0, 0, Q_0]
    ],
    #modeltype 1
    [
        [Q_0, 0, 0, 0, 0],
        [0, Q_0, 0, 0, 0],
        [0, 0, Q_0 / 1e2, 0, 0],
        [0, 0, 0, Q_0, 0],
        [0, 0, 0, 0, Q_0 / 1e8]        

    ],
    #modeltype 2
    [],
]

for i,noiseCov in enumerate(MeasurementNoiseCovs):
    MeasurementNoiseCovs[i] = np.array(noiseCov, dtype = float)
for i, noiseCov in enumerate(ProcessNoiseCovs):
    ProcessNoiseCovs[i] = np.array(noiseCov, dtype = float)

InitialStartCovs_withoutTimeDivision = [
    #modeltype 0
    [
        [MeasurementNoiseCovs[0][0][0], 0, MeasurementNoiseCovs[0][0][0], 0],
        [0, MeasurementNoiseCovs[0][1][1], 0, MeasurementNoiseCovs[0][1][1]],
        [MeasurementNoiseCovs[0][0][0], 0, 2*MeasurementNoiseCovs[0][0][0], 0],
        [0, MeasurementNoiseCovs[0][1][1], 0, 2*MeasurementNoiseCovs[0][1][1]]
    ],
    #modeltype1
    [
        [1.99997780e+00, 2.49414716e-04, 1.73227766e-04, 3.88280668e-04,
                0],
        [2.49414716e-04, 1.99980348e+00, 6.09740973e-05, 3.55028270e-04,
            0],
        [1.73227766e-04, 6.09740973e-05, 8.22416384e-01, 5.66820017e-04,
            0],
        [3.88280668e-04, 3.55028270e-04, 5.66820017e-04, 7.99940698e+00,
            0],
        [0, 0, 0, 0,
            0]
    ],
    #modeltype2
    [
        [1.99997780e+00, 2.49414716e-04, 1.73227766e-04, 3.88280668e-04,
                2.97819194e-04],
        [2.49414716e-04, 1.99980348e+00, 6.09740973e-05, 3.55028270e-04,
            1.49555161e-04],
        [1.73227766e-04, 6.09740973e-05, 8.22416384e-01, 5.66820017e-04,
            7.52457415e-01],
        [3.88280668e-04, 3.55028270e-04, 5.66820017e-04, 7.99940698e+00,
            6.98653089e-04],
        [2.97819194e-04, 1.49555161e-04, 7.52457415e-01, 6.98653089e-04,
            1.50502254e+00]        
    ]

]


#############
# Model 0
#############
def f_predict_model0(x, P, dt):

    F = np.array([      [1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]       ], dtype = float)

    x_predict = np.dot(F, x)
    P_predict = np.dot(F, np.dot(P, F.T)) + ProcessNoiseCovs[0]
    
    
    return (x_predict, P_predict)

def h_measure_model0(x):
    return (x[0:2], np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ]))

############
# Model 1
############
def f_predict_model1(x_, dt):

    x = np.copy(x_)
    
    x[2] = putAngleInRange(x[2])
    
    X_new = np.copy(x)

    x_new = x[0] + x[3] * dt * np.sin(x[2])
    y_new = x[1] + x[3] * dt * np.cos(x[2])
    
    X_new[0] = x_new
    X_new[1] = y_new

    return X_new

def h_measure_model1(x):
    return x[0:2]

############
# Model 2
############
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

def h_measure_model2(x):
    return x[0:2]

###########

class Track(object):

    def __init__(self, x0, P0):
        self.x = x0
        self.P = P0

class Tracker_SingleTarget_SingleModel_allMe(object):
  
    def __init__(self, modelType):
        """
        Initialises a tracker using initial bounding box.
        """
                
        self.modelType = modelType         
            
        self.track = None

        self.measurements = []

        self.unscentedWeights = None

        if(modelType == 1 or modelType == 2):
            self.Ws, self.Wc, self.lambda_ = uH.generateUnscentedWeights(L = 5, alpha = 1e-3, beta = 2, kappa = 0)

    def feedMeasurement(self, measurement, dt):

        self.measurements.append(measurement)

        if(self.track is None and len(self.measurements) > 1):

            #init the track
            if(self.modelType == 0): #linearModel, constant velocity, constant direction
                dx = (self.measurements[-1][0] - self.measurements[-2][0]) / dt
                dy = (self.measurements[-1][1] - self.measurements[-2][1]) / dt
                x0 = np.array([self.measurements[-1][0], self.measurements[-1][1] , dx, dy]).reshape((4,1))

                P0 = np.array(InitialStartCovs_withoutTimeDivision[0]) * [[1,1,1/dt,1],[1,1,1,1/dt],[1/dt,1,1/(dt**2),1],[1,1/dt,1,1/(dt**2)]]
                P0 = massageToCovariance(P0, 1e-6)

                self.track = Track(x0, P0)
                self.track.z, _ = h_measure_model0(self.track.x)
                
            elif(self.modelType == 1): #nonLinearModel, constant velocity, constant direction     
            
                dx = (self.measurements[-1][0] - self.measurements[-2][0]) / dt
                dy = (self.measurements[-1][1] - self.measurements[-2][1]) / dt
                phi = np.arctan(dy / dx)
                vel = np.sqrt(dx**2, dy**2)
                dphi = 0

                x0 = np.array([self.measurements[-1][0], self.measurements[-1][1], phi, vel, dphi]).reshape((5,1))

                P0 = np.array(InitialStartCovs_withoutTimeDivision[1]) * \
                    [[1,1,1,1/dt,1/dt],
                    [1,1,1,1/dt,1/dt],
                    [1,1,1,1/dt,1/dt],
                    [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)],
                    [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)]]

                P0 = massageToCovariance(P0, 1e-6)

                self.track = Track(x0, P0)
                self.track.z = h_measure_model1(self.track.x)                
                
            elif(self.modelType == 1): #nonLinearModel, constant veloctiy, contant turn rate
                if(len(self.measurements) > 2):
                    
                    dx1 = (self.measurements[-2][0] - self.measurements[-3][0]) / dt
                    dy1 = (self.measurements[-2][1] - self.measurements[-3][1]) / dt     
                    
                    dx2 = (self.measurements[-1][0] - self.measurements[-2][0]) / dt
                    dy2 = (self.measurements[-1][1] - self.measurements[-2][1]) / dt
                    
                    phi1 = np.arctan(dy1 / dx1)
                    phi2 = np.arctan(dy2 / dx2)
                    
                    vel = np.sqrt(dx2**2, dy2**2)
                    dphi = phi2 - phi1

                    x0 = np.array([self.measurements[-1][0], self.measurements[-1][1], phi, vel, dphi]).reshape((5,1))

                    P0 = np.array(InitialStartCovs_withoutTimeDivision[1]) * \
                        [[1,1,1,1/dt,1/dt],
                        [1,1,1,1/dt,1/dt],
                        [1,1,1,1/dt,1/dt],
                        [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)],
                        [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)]]

                    P0 = massageToCovariance(P0, 1e-6)


                    self.track = Track(x0, P0)
                    self.track.z = h_measure_model2(self.track.x)                                     
                        

        elif(self.track is not None):

            if(self.modelType == 0):
                #prediction
                self.track.x_predict, self.track.P_predict = f_predict_model0(self.track.x, self.track.P, dt)
                
                #update
                z_predict, H = h_measure_model0(self.track.x_predict)
                diff = (measurement - z_predict)
                S = np.dot(H, np.dot(self.track.P_predict, H.T)) + MeasurementNoiseCovs[0]
                kalmanGain = np.dot(self.track.P_predict, np.dot(H.T, np.linalg.inv(S)))

                self.track.x = self.track.x_predict + np.dot(kalmanGain, diff)
                temp = np.eye(self.track.x.shape[0]) - np.dot(kalmanGain, H)
                self.track.P = np.dot(temp, np.dot(self.track.P_predict, temp.T)) + np.dot(kalmanGain, np.dot(MeasurementNoiseCovs[0], kalmanGain.T))
                self.track.z, _ = h_measure_model0(self.track.x)


            elif(self.modelType == 1):
                #make sure cov matrix is symetric
                self.track.P = massageToCovariance(self.track.P, 1e-6)
                #generate sigma points and then predict
                sigmaPoints = uH.generateSigmaPoints(self.track.x, self.track.P, self.lambda_)
                self.track.x_predict, self.track.P_predict, Pzz, predictedMeasureMean, kalmanGain = uH.calculatePredictedState(f_predict_model1, dt, h_measure_model1, sigmaPoints, self.Ws, self.Wc, ProcessNoiseCovs[1], MeasurementNoiseCovs[1])

                #update
                diff = measurement - predictedMeasureMean
                self.track.x = self.track.x_predict + np.dot(kalmanGain, diff)
                self.track.P = self.track.P_predict - np.dot(kalmanGain, np.dot(Pzz, kalmanGain.T))
                self.track.z = h_measure_model1(self.track.x)


            elif(self.modelType == 2):
                print("todo modeltype 2")