# -*- coding: utf-8 -*-

import numpy as np


measurementNoiseStd = 2
Q_0 = 0.005

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
    [],
    #modeltype 2
    [],
]

for i,noiseCov in enumerate(MeasurementNoiseCovs):
    MeasurementNoiseCovs[i] = np.array(noiseCov, dtype = float)
for i, noiseCov in enumerate(ProcessNoiseCovs):
    ProcessNoiseCovs[i] = np.array(noiseCov, dtype = float)


#############
# Linear model
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

    def feedMeasurement(self, measurement, dt):

        self.measurements.append(measurement)

        if(self.track is None and len(self.measurements) > 1):

            #init the track
            if(self.modelType == 0): #linearModel, constant velocity, constant direction
                dx = (self.measurements[-1][0] - self.measurements[-2][0]) / dt
                dy = (self.measurements[-1][1] - self.measurements[-2][1]) / dt
                x0 = [self.measurements[-1][0], self.measurements[-1][0] , dx, dy]

                R1 = MeasurementNoiseCovs[0][0][0]
                R2 = MeasurementNoiseCovs[0][1][1]

                P0 = [
                    [R1, 0, R1/dt, 0],
                    [0, R2, 0, R2/dt],
                    [R1/dt, 0, 2*R1/(dt*dt), 0],
                    [0, R2/dt, 0, 2*R2/(dt*dt)]
                ]
                self.track = Track(x0, P0)
                self.track.z, _ = h_measure_model0(self.track.x)
                
            elif(self.modelType == 1): #nonLinearModel, constant velocity, constant direction     
            
                dx = (self.measurements[-1][0] - self.measurements[-2][0]) / dt
                dy = (self.measurements[-1][1] - self.measurements[-2][1]) / dt
                phi = np.arctan(dy / dx)
                vel = np.sqrt(dx**2, dy**2)
                dphi = 0
                
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
                print("todo modeltype 1")
            elif(self.modelType == 2):
                print("todo modeltype 2")