# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import torch

import sys
sys.path.append("../")


#variables
dtype_torch = torch.float64

measurementNoiseStd = np.sqrt(2)

MeasurementNoiseCov = \
        torch.tensor([[measurementNoiseStd ** 2, 0],
         [0, measurementNoiseStd **2]], dtype = dtype_torch)
    

InitialStartCov_withoutTimeDivision = \
    torch.tensor(    [
        [MeasurementNoiseCov[0][0], 0, MeasurementNoiseCov[0][0], 0],
        [0, MeasurementNoiseCov[1][1], 0, MeasurementNoiseCov[1][1]],
        [MeasurementNoiseCov[0][0], 0, 2*MeasurementNoiseCov[0][0], 0],
        [0, MeasurementNoiseCov[1][1], 0, 2*MeasurementNoiseCov[1][1]]
    ], dtype=dtype_torch)



#to optimize
ProcessNoiseCov = None


#helpers


def massageToCovariance(P, scale):
    return 1/2*(P + P.T) + torch.eye(P.shape[0], dtype=dtype_torch) * scale

#models

def f_predict_model0(x, P, dt):

    F = torch.tensor([      [1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]       ], dtype = dtype_torch)    


    x_predict = torch.mm(F, x)
    P_predict = torch.mm(F, torch.mm(P, F.T)) + ProcessNoiseCov
    
    
    return (x_predict, P_predict)

def h_measure_model0(x):
    return (x[0:2], torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=dtype_torch))






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
        

class Tracker_SingleTarget_SingleModel_Linear_allMe(object):
  
    def __init__(self):
                
            
        self.track = None

        self.measurements = []
        

    
    def predict(self, dt):

        if(self.track is not None):
                       
            self.track.x_predict, self.track.P_predict = f_predict_model0(self.track.x, self.track.P, dt)
            self.track.z_predict, H = h_measure_model0(self.track.x_predict)
            self.track.S = torch.mm(H, torch.mm(self.track.P_predict, H.T)) + MeasurementNoiseCov
            self.track.kalmanGain = torch.mm(self.track.P_predict, torch.mm(H.T, torch.inverse(self.track.S)))            
            

    def detachTrack(self):
        
        self.track.x = self.track.x.detach()
        self.track.P = self.track.P.detach()
      



    def feedMeasurement(self, measurement, dt):

        self.measurements.append(measurement)

        if(self.track is None and len(self.measurements) > 1):

            dx = (self.measurements[-1][0] - self.measurements[-2][0]) / dt
            dy = (self.measurements[-1][1] - self.measurements[-2][1]) / dt
            x0 = torch.tensor([self.measurements[-1][0], self.measurements[-1][1] , dx, dy] ,dtype=dtype_torch).reshape((4,1))

            P0 = torch.tensor(InitialStartCov_withoutTimeDivision , dtype=dtype_torch) * \
                torch.tensor(
                    [[1,1,1/dt,1],
                    [1,1,1,1/dt],
                    [1/dt,1,1/(dt**2),1],
                    [1,1/dt,1,1/(dt**2)]]
                ,dtype = dtype_torch)
                
            P0 = massageToCovariance(P0, 1e-6)

            self.track = Track(x0, P0)
            self.track.z, _ = h_measure_model0(self.track.x)         


        elif(self.track is not None):


            #make sure cov matrix is symetric
            self.track.P = massageToCovariance(self.track.P, 1e-8)

            #predict
            self.predict(dt)

            #update
            diff = measurement - self.track.z_predict
            self.track.x = self.track.x_predict + torch.mm(self.track.kalmanGain, diff)
            self.track.P = self.track.P_predict - torch.mm(self.track.kalmanGain, torch.mm(self.track.S, self.track.kalmanGain.T))
            self.track.z, _ = h_measure_model0(self.track.x)
            
            return self.track.z
        
        
        
        
        
        
        
        
        