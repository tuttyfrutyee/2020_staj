# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

from filterpy.kalman import KalmanFilter
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints

import numpy as np


def f_unscented_turnRateModel(x_, dt):
    
    x = np.copy(x_)
    
    x[2] = putAngleInRange(x[2])
    x[4] = putAngleInRange(x[4])
    
    X_new = np.copy(x)

    
    
    if(dt * x[4] < np.pi /2):
        
        x_new = x[0] + x[3] / x[4] * ( -np.sin(x[2]) + np.sin( x[2] + dt * x[4] ) )
        y_new = x[1] + x[3] / x[4] * ( np.cos(x[2])  - np.cos( x[2] + dt * x[4] ) )
        
        phi_new = x[2] + dt * x[4] 
        
    else:
        
        x_new = x[0] + x[3] / x[4] * ( -np.sin(x[2])  )
        y_new = x[1] + x[3] / x[4] * ( np.cos(x[2])  )
        
        phi_new = x[2]       
    

    phi_new = putAngleInRange(phi_new)
    
    X_new[0] = x_new
    X_new[1] = y_new
    X_new[2] = phi_new
    
    return X_new

def putAngleInRange(angle):
    
    angle = angle % np.pi
    
    if(angle > (np.pi/2)):
        angle -= np.pi
    elif(angle < (-np.pi/2)):
        angle += np.pi
        
    return angle
    
    
def h_unscented_turnRateModel(x):
    return x[0:2]
    


class Tracker_SingleTarget_LinearSingleModel(object):
  """
  This class tracks object point(single target) with a linear model assumption
  """
  
  def __init__(self, modelType, deltaT, measurementNoiseStd ):
    """
    Initialises a tracker using initial bounding box.
    """
    
    if(modelType == 0): #use contant linear velocity model
        
        # x = [x, y, ax, ay]
        
        self.updatedPredictions = []
        
        self.kf = KalmanFilter(dim_x = 4, dim_z = 2)
        
        self.kf.F = np.array([
                [1, 0, deltaT, 0],
                [0, 1, 0, deltaT],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
            ])
        
        self.kf.x = np.array([0.01, 0.01, 0.01, 0.01])
        self.kf.P *= measurementNoiseStd**2
        self.kf.Q *= 0.005
        self.kf.R *= measurementNoiseStd**2
        
    elif(modelType == 1): 
        """
            Use constant turn rate, constant linear velocity
        use unscented kalman filter        
        """
        
        points = MerweScaledSigmaPoints(5, alpha=0.0025, beta=2., kappa=0)
        
        self.updatedPredictions = []        
        
        self.kf = UnscentedKalmanFilter(dim_x=5, dim_z=2, dt=deltaT, fx=f_unscented_turnRateModel, hx=h_unscented_turnRateModel, points=points)
        
        self.kf.x = np.array([0.01, 0.01, 0.01, 0.01, 0.001])
        
        self.kf.P = np.eye(5) * (measurementNoiseStd**2)
        
        self.kf.R = np.eye(2) * (measurementNoiseStd**2)
        
        self.kf.Q = np.diag([1e-3, 1e-3, 1e-4, 4e-3, 8e-14])
                
        
  def predictAndUpdate(self, measurement):
        
    self.kf.predict()
    self.kf.update(measurement)
    
    self.updatedPredictions.append(np.array(( self.kf.x_post[0], self.kf.x_post[1] )) )
    
