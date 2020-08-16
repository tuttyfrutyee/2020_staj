# -*- coding: utf-8 -*-

import numpy as np
import torch

dtype_numpy = np.float
dtype_torch = torch.float64

measurementNoiseStd = np.sqrt(2)

processNoise = None

def massageToCovariance(P, scale):
    return 1/2*(P + P.T) + torch.eye(P.shape[0]) * scale


def generateRandomCovariance_positiveDefinite(dim):
    randomCovariance = np.random.randn(dim, dim)
    randomCovariance = 0.5*(randomCovariance + randomCovariance.T) / ( np.max(abs(randomCovariance))) + dim * np.eye(dim)
    return torch.tensor(randomCovariance)

def calculateLoss(groundTruth, predictionMeasured, S):
    
    diff = groundTruth - predictionMeasured
    
    #return  torch.log(torch.det(S)) + torch.mm(diff.T, torch.mm(torch.inverse(S), diff))
    return torch.mm(diff.T, diff)

########


MeasurementNoiseCov = [
    
        [measurementNoiseStd * measurementNoiseStd, 0],
        [0, measurementNoiseStd * measurementNoiseStd]

]

MeasurementNoiseCov = torch.tensor(MeasurementNoiseCov, dtype=dtype_torch)

    

InitialStartCov_withoutTimeDivision = [

        [MeasurementNoiseCov[0][0], 0, MeasurementNoiseCov[0][0], 0],
        [0, MeasurementNoiseCov[1][1], 0, MeasurementNoiseCov[1][1]],
        [MeasurementNoiseCov[0][0], 0, 2*MeasurementNoiseCov[0][0], 0],
        [0, MeasurementNoiseCov[1][1], 0, 2*MeasurementNoiseCov[1][1]]
]

InitialStartCov_withoutTimeDivision = torch.tensor(InitialStartCov_withoutTimeDivision, dtype=dtype_torch)


#############
# Model 0
#############
def f_predict_model0(x, P, dt):

    F = np.array([      [1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]       ], dtype = dtype_numpy)    

    F = torch.from_numpy(F)           

    x_predict = torch.mm(F, x)
    P_predict = torch.mm(F, torch.mm(P, F.T)) + processNoise
    
    
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

class Tracker_SingleTarget_SingleModel_allMe(object):
  
    def __init__(self, modelType):
                
        self.modelType = modelType         
            
        self.track = None

        self.measurements = []
                
        self.loss = 0
        
        self.losses = []
        

    
    def predict(self, dt):

        if(self.track):

            if(self.modelType == 0):

                self.track.x_predict, self.track.P_predict = f_predict_model0(self.track.x, self.track.P, dt)


    def feedMoment(self, moment, dt, endSequence):

        measurement = moment[0:2].reshape((2,1))
        groundTruth = moment[2:4].reshape((2,1))
        
        self.measurements.append(measurement)

        if(self.track is None and len(self.measurements) > 1):

            #init the track
            if(self.modelType == 0): #linearModel, constant velocity, constant direction
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

            if(self.modelType == 0):
                #prediction
                self.predict(dt)                
                #update
                z_predict, H = h_measure_model0(self.track.x_predict)
                diff = (measurement - z_predict)
                self.track.S = torch.mm(H, torch.mm(self.track.P_predict, H.T)) + MeasurementNoiseCov
                kalmanGain = torch.mm(self.track.P_predict, torch.mm(H.T, torch.inverse(self.track.S)))


                self.track.x = self.track.x_predict + torch.mm(kalmanGain, diff)
                temp = torch.eye(self.track.x.shape[0], dtype=dtype_torch) - torch.mm(kalmanGain, H)
                self.track.P = torch.mm(temp, torch.mm(self.track.P_predict, temp.T)) + torch.mm(kalmanGain, torch.mm(MeasurementNoiseCov, kalmanGain.T))
                self.track.z, _ = h_measure_model0(self.track.x)
                
                a = torch.mm(H, torch.mm(self.track.P, H.T))
                self.loss += calculateLoss(groundTruth, self.track.z, a)
                
                if(endSequence):

                    self.loss.backward()
                    self.losses.append(self.loss.item())
                    self.loss = 0
                    
                    self.track.x = self.track.x.detach()
                    self.track.P = self.track.P.detach()
                    self.track.z = self.track.z.detach()
                    self.track.S = self.track.S.detach()

                    


