# -*- coding: utf-8 -*-

import numpy as np
import myHelpers.unscentedHelper as uH


measurementNoiseStd = np.sqrt(2)
Q_0 = 0.008

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
    #modeltype 3
    [
        [measurementNoiseStd * measurementNoiseStd, 0],
        [0, measurementNoiseStd * measurementNoiseStd]
    ],    
]

ProcessNoiseCovs = [
    #modeltype 0

    [
        [ 0.0251,  0.0219, -0.0072, -0.0054],
        [ 0.0219,  0.0199, -0.0064, -0.0049],
        [-0.0072, -0.0064,  0.0023,  0.0016],
        [-0.0054, -0.0049,  0.0016,  0.0017]
        
        # [Q_0, 0, 0, 0],
        # [0, Q_0, 0, 0],
        # [0, 0, Q_0 , 0],
        # [0, 0, 0, Q_0],    
     
         # [0,0,0,0],
         # [0,0,0,0],
         # [0,0,0,0],
         # [0,0,0,0]     

    ],
    #modeltype 1
    (np.array([
       [0.12958419304911287,0,0,0,0],
       [0,0.20416385918814656,0,0,0],
       [0,0,0.008794949000079913,0,0],
       [0,0,0,0.8057826337426066,0],
       [0,0,0,0,0] 
     
        # [Q_0, 0, 0, 0, 0],
        # [0, Q_0, 0, 0, 0],
        # [0, 0, Q_0 / 1e2, 0, 0],
        # [0, 0, 0, Q_0, 0],
        # [0, 0, 0, 0, 0]       
     
      # [0,0,0,0,0],
      # [0,0,0,0,0],
      # [0,0,0,0,0],
      # [0,0,0,0,0],
      # [0,0,0,0,0]
    ]) / 200).tolist(),
    #modeltype 2
    (np.array([
        
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
         
     ])/ 20000).tolist(),
    #modeltype 3
    [
        [1e-2, 0, 0, 0, 0],
        [0, 1e-2, 0, 0, 0],
        [0, 0, 0 , 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0 ]             
     ],     
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
    ],

    #modeltype3
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

    x_new = x[0] + x[3] * dt * np.cos(x[2])
    y_new = x[1] + x[3] * dt * np.sin(x[2])
    
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
# Model 3
###########

def f_predict_model3(x_, dt):
    return x_

def h_measure_model3(x):
    return x[0:2]

##########

class Track(object):

    def __init__(self, x0, P0):
        self.x = x0
        self.P = P0
        self.x_predict = None
        self.P_predict = None
        self.z_predict = None
        self.S = None
        self.kalmanGain = None
        #optional
        self.H = None #only for linear kalman filter

class Tracker_SingleTarget_SingleModel_allMe(object):
  
    def __init__(self, modelType):
                
        self.modelType = modelType         
            
        self.track = None

        self.measurements = []
        
        if(modelType == 1 or modelType == 2 or modelType == 3):
            self.Ws, self.Wc, self.lambda_ = uH.generateUnscentedWeights(L = 5, alpha = 1e-3, beta = 2, kappa = 0)

    def putMeasurement(self, measurement):
        #can be used to put only measurement to self.measurements without state update
        self.measurements.append(measurement)
    
    def predict(self, dt):

        if(self.track):

            if(self.modelType == 0):

                self.track.x_predict, self.track.P_predict = f_predict_model0(self.track.x, self.track.P, dt)
                self.track.z_predict, self.track.H = h_measure_model0(self.track.x_predict)
                self.track.S = np.dot(self.track.H, np.dot(self.track.P_predict, self.track.H.T)) + MeasurementNoiseCovs[0]
                self.track.kalmanGain = np.dot(self.track.P_predict, np.dot(self.track.H.T, np.linalg.inv(self.track.S)))                


            elif(self.modelType == 1 or self.modelType == 2 or self.modelType == 3):

                self.track.P = massageToCovariance(self.track.P, 1e-8)

                sigmaPoints = uH.generateSigmaPoints(self.track.x, self.track.P, self.lambda_)

                if(self.modelType == 1):

                    self.track.x_predict, self.track.P_predict = uH.predictNextState(f_predict_model1, dt, sigmaPoints, self.Ws, self.Wc, ProcessNoiseCovs[1])                    
            

                    #sigmaPoints = uH.generateSigmaPoints(self.track.x_predict, self.track.P_predict, self.lambda_)                                       
                    #self.track.S, self.track.kalmanGain, self.track.z_predict = uH.calculateUpdateParameters(self.track.x_predict, self.track.P_predict, h_measure_model1, sigmaPoints, self.Ws, self.Wc, MeasurementNoiseCovs[1] ) 
                    
                    H = np.array([[1, 0, 0, 0, 0],[0, 1, 0, 0, 0]])
                    self.track.S = np.dot(H, np.dot(self.track.P_predict, H.T)) + MeasurementNoiseCovs[1]
                    self.track.kalmanGain = np.dot(self.track.P_predict, np.dot(H.T, np.linalg.inv(self.track.S)))
                    self.track.z_predict = h_measure_model1(self.track.x_predict)

                elif(self.modelType == 2):

                    self.track.x_predict, self.track.P_predict = uH.predictNextState(f_predict_model2, dt, sigmaPoints, self.Ws, self.Wc, ProcessNoiseCovs[2])
                    sigmaPoints = uH.generateSigmaPoints(self.track.x_predict, self.track.P_predict, self.lambda_)                    
                    self.track.S, self.track.kalmanGain, self.track.z_predict = uH.calculateUpdateParameters(self.track.x_predict, self.track.P_predict, h_measure_model2, sigmaPoints, self.Ws, self.Wc, MeasurementNoiseCovs[2] ) 
                
                elif(self.modelType == 3):
                    
                    self.track.x_predict, self.track.P_predict = uH.predictNextState(f_predict_model3, dt, sigmaPoints, self.Ws, self.Wc, ProcessNoiseCovs[3])
                    sigmaPoints = uH.generateSigmaPoints(self.track.x_predict, self.track.P_predict, self.lambda_)                    
                    self.track.S, self.track.kalmanGain, self.track.z_predict = uH.calculateUpdateParameters(self.track.x_predict, self.track.P_predict, h_measure_model3, sigmaPoints, self.Ws, self.Wc, MeasurementNoiseCovs[3] ) 



    def feedMeasurement(self, measurement, dt):

        self.measurements.append(measurement)

        if(self.track is None and len(self.measurements) > 1):

            #init the track
            if(self.modelType == 0): #linearModel, constant velocity, constant direction
                dx = (self.measurements[-1][0] - self.measurements[-2][0]) / dt
                dy = (self.measurements[-1][1] - self.measurements[-2][1]) / dt
                x0 = np.array([self.measurements[-1][0], self.measurements[-1][1] , dx, dy]).reshape((4,1))

                P0 = np.array(InitialStartCovs_withoutTimeDivision[0]) * \
                    [[1,1,1/dt,1],
                    [1,1,1,1/dt],
                    [1/dt,1,1/(dt**2),1],
                    [1,1/dt,1,1/(dt**2)]]
                    
                P0 = massageToCovariance(P0, 1e-6)

                self.track = Track(x0, P0)
                self.track.z, _ = h_measure_model0(self.track.x)
                
            elif(self.modelType == 1): #nonLinearModel, Constant Velocity (CV) Model   
            
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
                
            elif(self.modelType == 2): #nonLinearModel, Constant Turn-Rate Velocity (CTRV) Model
                if(len(self.measurements) > 2):
                    
                    dx1 = (self.measurements[-2][0] - self.measurements[-3][0]) / dt
                    dy1 = (self.measurements[-2][1] - self.measurements[-3][1]) / dt     
                    
                    dx2 = (self.measurements[-1][0] - self.measurements[-2][0]) / dt
                    dy2 = (self.measurements[-1][1] - self.measurements[-2][1]) / dt
                    
                    phi1 = np.arctan(dy1 / dx1)
                    phi2 = np.arctan(dy2 / dx2)
                    
                    vel = np.sqrt(dx2**2, dy2**2)
                    dphi = phi2 - phi1

                    x0 = np.array([self.measurements[-1][0], self.measurements[-1][1], phi2, vel, dphi]).reshape((5,1))

                    P0 = np.array(InitialStartCovs_withoutTimeDivision[2]) * \
                        [[1,1,1,1/dt,1/dt],
                        [1,1,1,1/dt,1/dt],
                        [1,1,1,1/dt,1/dt],
                        [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)],
                        [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)]]

                    P0 = massageToCovariance(P0, 1e-6)


                    self.track = Track(x0, P0)
                    self.track.z = h_measure_model2(self.track.x)

            elif(self.modelType == 3): # Random Motion (RM) Model 

                if(len(self.measurements) > 2):
                    
                    dx1 = (self.measurements[-2][0] - self.measurements[-3][0]) / dt
                    dy1 = (self.measurements[-2][1] - self.measurements[-3][1]) / dt     
                    
                    dx2 = (self.measurements[-1][0] - self.measurements[-2][0]) / dt
                    dy2 = (self.measurements[-1][1] - self.measurements[-2][1]) / dt
                    
                    phi1 = np.arctan(dy1 / dx1)
                    phi2 = np.arctan(dy2 / dx2)
                    
                    vel = np.sqrt(dx2**2, dy2**2)
                    dphi = phi2 - phi1

                    x0 = np.array([self.measurements[-1][0], self.measurements[-1][1], phi2, vel, dphi]).reshape((5,1))

                    P0 = np.array(InitialStartCovs_withoutTimeDivision[3]) * \
                        [[1,1,1,1/dt,1/dt],
                        [1,1,1,1/dt,1/dt],
                        [1,1,1,1/dt,1/dt],
                        [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)],
                        [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)]]

                    P0 = massageToCovariance(P0, 1e-6)


                    self.track = Track(x0, P0)
                    self.track.z = h_measure_model3(self.track.x)                    
                        

        elif(self.track is not None):

            if(self.modelType == 0):

                #prediction
                self.predict(dt)           

                #update
                diff = (measurement - self.track.z_predict)
                self.track.x = self.track.x_predict + np.dot(self.track.kalmanGain, diff)
                temp = np.eye(self.track.x.shape[0]) - np.dot(self.track.kalmanGain, self.track.H)
                self.track.P = np.dot(temp, np.dot(self.track.P_predict, temp.T)) + np.dot(self.track.kalmanGain, np.dot(MeasurementNoiseCovs[0], self.track.kalmanGain.T))
                self.track.z, _ = h_measure_model0(self.track.x)


            elif(self.modelType == 1):

                #make sure cov matrix is symetric
                self.track.P = massageToCovariance(self.track.P, 1e-8)

                #predict
                self.predict(dt)

                #update
                diff = measurement - self.track.z_predict
                self.track.x = self.track.x_predict + np.dot(self.track.kalmanGain, diff)
                self.track.P = self.track.P_predict - np.dot(self.track.kalmanGain, np.dot(self.track.S, self.track.kalmanGain.T))
                self.track.z = h_measure_model1(self.track.x)


            elif(self.modelType == 2):

                #make sure cov matrix is symetric
                self.track.P = massageToCovariance(self.track.P, 1e-8)

                #predict
                self.predict(dt)

                #update
                diff = measurement - self.track.z_predict
                self.track.x = self.track.x_predict + np.dot(self.track.kalmanGain, diff)
                self.track.P = self.track.P_predict - np.dot(self.track.kalmanGain, np.dot(self.track.S, self.track.kalmanGain.T))
                self.track.z = h_measure_model1(self.track.x)
                
            elif(self.modelType == 3):

                #make sure cov matrix is symetric
                self.track.P = massageToCovariance(self.track.P, 1e-8)

                #predict
                self.predict(dt)
                
                #update
                diff = measurement - self.track.z_predict
                self.track.x = self.track.x_predict + np.dot(self.track.kalmanGain, diff)
                self.track.P = self.track.P_predict - np.dot(self.track.kalmanGain, np.dot(self.track.S, self.track.kalmanGain.T))
                self.track.z = h_measure_model1(self.track.x)    
                
                
                
                