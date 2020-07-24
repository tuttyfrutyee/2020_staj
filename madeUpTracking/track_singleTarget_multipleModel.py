# -*- coding: utf-8 -*-

from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import IMMEstimator

import numpy as np

maxChange = 0

def f_unscented_turnRateModel(x_, dt):
    
    global maxChange
    
    x = np.copy(x_)
    
    x[2] = putAngleInRange(x[2])
    x[4] = putAngleInRange(x[4])
    
    X_new = np.copy(x)

    
    maxChange = max(maxChange, dt*x[4])
    
    if(dt * x[4] < np.pi / 4):
        
        x_new = x[0] + x[3] / x[4] * ( -np.sin(x[2]) + np.sin( x[2] + dt * x[4] ) )
        y_new = x[1] + x[3] / x[4] * ( np.cos(x[2])  - np.cos( x[2] + dt * x[4] ) )
        
        phi_new = x[2] + dt * x[4] 
        
    else:
        
        print("cutted")
        
        x_new = x[0] + x[3] / x[4] * ( -np.sin(x[2])  )
        y_new = x[1] + x[3] / x[4] * ( np.cos(x[2])  )
        
        phi_new = x[2]       
    

    phi_new = putAngleInRange(phi_new)
    
    X_new[0] = x_new
    X_new[1] = y_new
    X_new[2] = phi_new
    
    return X_new

def f_unscented_linearModel(x_, dt):
    
    x = np.copy(x_)
    
    x[2] = putAngleInRange(x[2])
    
    X_new = np.copy(x)

    x_new = x[0] + x[3] * dt * np.sin(x[2])
    y_new = x[1] + x[3] * dt * np.cos(x[2])
    
    X_new[0] = x_new
    X_new[1] = y_new

    return X_new

def f_unscented_randomModel(x_, dt):
    
    return x_



def putAngleInRange(angle):
    
    angle = angle % (2*np.pi)
    
    if(angle > (np.pi)):
        angle -= 2*np.pi
    elif(angle < (-np.pi)):
        angle += 2*np.pi
        
    return angle
    
    
def h_unscented_turnRateModel(x):
    return x[0:2]
    
def h_unscented_linearModel(x):
    return x[0:2]

def h_unscented_randomModel(x):
    return x[0:2]


class Tracker_SingleTarget_MultipleModel(object):
  """
  """
  
  def __init__(self, deltaT, measurementNoiseStd ):
      
    self.updatedPredictions = []
    self.mus = []
      

    """
        First init constant linear model
    """
    
    points1 = MerweScaledSigmaPoints(5, alpha=0.0025, beta=2., kappa=0)
         
    
    self.constantLinearModel = UnscentedKalmanFilter(dim_x=5, dim_z=2, dt=deltaT, fx=f_unscented_linearModel, hx=h_unscented_linearModel, points=points1)

    self.constantLinearModel.x = np.array([0.01, 0.01, 0.01, 0.01, 0])
    
    self.constantLinearModel.P = np.eye(5) * (measurementNoiseStd**2) / 2.0
    
    self.constantLinearModel.R = np.eye(2) * (measurementNoiseStd**2)
    
    self.constantLinearModel.Q = np.diag([0.003, 0.003, 6e-4, 0.004, 0])   
    
    """
        Second init constant turn rate model
    """
    
    points2 = MerweScaledSigmaPoints(5, alpha=0.0025, beta=2., kappa=0)
         
    
    self.constantTurnRateModel = UnscentedKalmanFilter(dim_x=5, dim_z=2, dt=deltaT, fx=f_unscented_turnRateModel, hx=h_unscented_turnRateModel, points=points2)

    self.constantTurnRateModel.x = np.array([0.01, 0.01, 0.01, 0.001, 1e-5])
    
    self.constantTurnRateModel.P = np.eye(5) * (measurementNoiseStd**2) / 2.0
    
    self.constantTurnRateModel.R = np.eye(2) * (measurementNoiseStd**2)
    
    self.constantTurnRateModel.Q = np.diag([1e-24, 1e-24, 1e-3, 4e-3,  1e-10])
    
    
    """
        Third init random motion model
    """
    points3 = MerweScaledSigmaPoints(5, alpha=0.0025, beta=2., kappa=0)
    
    self.randomModel = UnscentedKalmanFilter(dim_x=5, dim_z=2, dt=deltaT, fx=f_unscented_randomModel, hx=h_unscented_randomModel, points=points3)

    self.randomModel.x = np.array([0.01, 0.01, 0.01, 0.001, 1e-5])
    
    self.randomModel.P = np.eye(5) * (measurementNoiseStd**2) / 2.0
    
    self.randomModel.R = np.eye(2) * (measurementNoiseStd**2)
    
    self.randomModel.Q = np.diag([1, 1, 1e-24, 1e-24,  1e-24])    
    
    #############################33
    
    if(1):
        
        filters = [self.constantLinearModel, self.constantTurnRateModel]
        
        
        mu = [0.5, 0.5]
        
        trans = np.array([[0.9, 0.1], [0.1, 0.9]])
        
        self.imm =  IMMEstimator(filters, mu, trans)        
        
    else:
        
        filters = [self.constantLinearModel, self.constantTurnRateModel, self.randomModel]
        
        
        mu = [0.34, 0.33, 0.33]
        
        trans = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        
        self.imm =  IMMEstimator(filters, mu, trans)
    
    
    
        
  def predictAndUpdate(self, measurement):
        
    self.imm.P = 1/2.0*(self.imm.P + self.imm.P.T)  
      
        
    self.imm.predict()
    self.imm.update(measurement)
    
    self.updatedPredictions.append(np.array(( self.imm.x_post[0], self.imm.x_post[1] )) )
    self.mus.append(self.imm.mu)























