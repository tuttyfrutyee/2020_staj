# -*- coding: utf-8 -*-

import numpy as np
import torch

import sys
sys.path.append("../")
import myHelpers.unscentedHelper as uH


#variables
dtype_torch = torch.float64

ProcessNoiseCov = None


measurementNoiseStd = 1.2

MeasurementNoiseCov = \
        torch.tensor([[measurementNoiseStd ** 2, 0],
         [0, measurementNoiseStd **2]], dtype = dtype_torch)
    



InitialStartCov_withoutTimeDivision = \
    torch.tensor([
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
    ], dtype=dtype_torch)




#helpers


def putAngleInRange(angle):
    
    angle = angle % (2*np.pi)
    
    if(angle > (np.pi)):
        angle -= 2*np.pi
    elif(angle < (-np.pi)):
        angle += 2*np.pi
        
    return angle

def massageToCovariance(P, scale):
    return 1/2*(P + P.T) + torch.eye(P.shape[0], dtype=dtype_torch) * scale


class normalizeState(torch.autograd.Function):
    
    @staticmethod
    def forward(self, state):
        self.save_for_backward(state)
        
        state_ = state.clone()
        
        state_[2] = putAngleInRange(state_[2])
        state_[4] = putAngleInRange(state_[4])
        
        return state_

    
    @staticmethod
    def backward(self, grad_output):
        
        input, = self.saved_tensors
        
        grad_input = grad_output.clone()
        
        return grad_input

stateNormalizer = normalizeState.apply

########



############
# Model 1
############
def f_predict_model2(x, dt):
    
    x_ = x.clone()
    x__ = x.clone()


    x_[0] = x__[0] + x__[3] / x__[4] * (-torch.sin(x__[2]) + torch.sin(x__[2] + dt * x__[4]) )    
    x_[1] = x__[1] + x__[3] / x__[4] * (torch.cos(x__[2]) - torch.cos(x__[2] + dt * x__[4]) )
    x_[2] = x__[2] + dt * x__[4]
    
            
    return x_

def h_measure_model2(x):
    x_ = x.clone()
    
    return x_[0:2]


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

class Tracker_SingleTarget_SingleModel_CTRV_allMe(object):
  
    def __init__(self):
                            
        self.track = None

        self.measurements = []
        
        self.Ws, self.Wc, self.lambda_ = uH.generateUnscentedWeights(L = 5, alpha = 1e-3, beta = 2, kappa = 0)

    
    def predict(self, dt):

        if(self.track is not None):
            

            self.track.P = massageToCovariance(self.track.P, 1e-8)

            sigmaPoints = uH.generateSigmaPoints(self.track.x, self.track.P, self.lambda_)
            

            self.track.x_predict, self.track.P_predict = uH.predictNextState(f_predict_model2, dt, sigmaPoints, self.Ws, self.Wc, ProcessNoiseCov)                    
            self.track.x_predict = stateNormalizer(self.track.x_predict)

            H = torch.tensor([[1, 0, 0, 0, 0],[0, 1, 0, 0, 0]], dtype=dtype_torch)
            self.track.S = torch.mm(H, torch.mm(self.track.P_predict, H.T)) + MeasurementNoiseCov
            self.track.kalmanGain = torch.mm(self.track.P_predict, torch.mm(H.T, torch.inverse(self.track.S)))
            self.track.z_predict = h_measure_model2(self.track.x_predict)

    def detachTrack(self):

        self.track.x = self.track.x.detach()
        self.track.P = self.track.P.detach()



    def feedMeasurement(self, measurement, dt):

        self.measurements.append(measurement)

        if(self.track is None and len(self.measurements) > 2):


                            
            dx1 = (self.measurements[-2][0] - self.measurements[-3][0]) / dt
            dy1 = (self.measurements[-2][1] - self.measurements[-3][1]) / dt     
            
            dx2 = (self.measurements[-1][0] - self.measurements[-2][0]) / dt
            dy2 = (self.measurements[-1][1] - self.measurements[-2][1]) / dt
            
            phi1 = np.arctan(dy1 / dx1)
            phi2 = np.arctan(dy2 / dx2)
            
            vel = np.sqrt(dx2**2 + dy2**2)
            dphi = phi2 - phi1

            x0 = torch.tensor([self.measurements[-1][0], self.measurements[-1][1], phi2, vel, dphi]).reshape((5,1))

            P0 = InitialStartCov_withoutTimeDivision * \
                torch.tensor([[1,1,1,1/dt,1/dt],
                [1,1,1,1/dt,1/dt],
                [1,1,1,1/dt,1/dt],
                [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)],
                [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)]], dtype=dtype_torch)

            P0 = massageToCovariance(P0, 1e-6)


            self.track = Track(x0, P0)
            self.track.z = h_measure_model2(self.track.x)             


        elif(self.track is not None):


            self.track.P = massageToCovariance(self.track.P, 1e-8)

            #predict
            self.predict(dt)

            #update
            diff = measurement - self.track.z_predict
            self.track.x = self.track.x_predict + torch.mm(self.track.kalmanGain, diff)
            self.track.x = stateNormalizer(self.track.x)            
            self.track.P = self.track.P_predict - torch.mm(self.track.kalmanGain, torch.mm(self.track.S, self.track.kalmanGain.T))
            self.track.z = h_measure_model2(self.track.x)
            
            
            return self.track.z