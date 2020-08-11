# -*- coding: utf-8 -*-

from filterpy.kalman import KalmanFilter
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints



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



class Tracker_SingleTarget_SingleModel_filterpy(object):
  """
  This class tracks object point(single target) with a linear model assumption
  """
  
  def __init__(self, modelType, deltaT, measurementNoiseStd ):
    """
    Initialises a tracker using initial bounding box.
    """
    
    self.measurementNoiseStd = measurementNoiseStd
    
    self.modelType = modelType
    
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
            Constant turn rate model
            
        """
        
        points1 = MerweScaledSigmaPoints(5, alpha=0.001, beta=2., kappa=0)
        
        self.updatedPredictions = []        
        
        self.kf = UnscentedKalmanFilter(dim_x=5, dim_z=2, dt=deltaT, fx=f_unscented_turnRateModel, hx=h_unscented_turnRateModel, points=points1)

        self.kf.x = np.array([1e-3, 1e-3, 1e-3, 1e-5, 1e-10])
        
        self.kf.P = np.eye(5) * (measurementNoiseStd**2)/2.0
        
        self.kf.R = np.eye(2) * (measurementNoiseStd**2) 
        
        self.kf.Q = np.diag([1e-24, 1e-24, 1e-3, 4e-3,  1e-10])
            
    elif(modelType == 2):
        """
            Constant linear velocity model
        """
        
        points1 = MerweScaledSigmaPoints(5, alpha=0.001, beta=2., kappa=0)
        
        
        self.updatedPredictions = []                
         
        self.kf = []        
    
        self.kf = UnscentedKalmanFilter(dim_x=5, dim_z=2, dt=deltaT, fx=f_unscented_linearModel, hx=h_unscented_linearModel, points=points1)
    
        self.kf.x = np.array([1e-3, 1e-3, 1e-3, 1e-5, 0])
        
        self.kf.P = np.eye(5) * (measurementNoiseStd**2) / 2.0
        
        self.kf.R = np.eye(2) * (measurementNoiseStd**2)  
        
        self.kf.Q = np.diag([0.003, 0.003, 6e-4, 0.004, 0])       
        
    elif(modelType == 3):
        """
            Random Motion Model
        """
        
        points1 = MerweScaledSigmaPoints(5, alpha=0.001, beta=2., kappa=0)
        
        
        self.updatedPredictions = []                
         
        self.kf = []        
    
        self.kf = UnscentedKalmanFilter(dim_x=5, dim_z=2, dt=deltaT, fx=f_unscented_randomModel, hx=h_unscented_randomModel, points=points1)
    
        self.kf.x = np.array([1e-3, 1e-3, 1e-3, 1e-5, 0])
        
        self.kf.P = np.eye(5) * (measurementNoiseStd**2) / 2.0
        
        self.kf.R = np.eye(2) * (measurementNoiseStd**2)  
        
        self.kf.Q = np.diag([1, 1, 1e-24, 1e-24,  1e-24])           
        
  def predictAndUpdate(self, measurement):
      

    self.kf.predict()
    self.kf.update(measurement)
    
    self.updatedPredictions.append(np.array(( self.kf.x_post[0], self.kf.x_post[1] )) )
    
