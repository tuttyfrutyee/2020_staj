# -*- coding: utf-8 -*-

import numpy as np
import myHelpers.unscentedHelper as uH

import myHelpers.immHelper as IMM_helper

# Will use single model to create different models
from Trackers.SingleTarget.allMe.track_singleTarget_singleModel import Tracker_SingleTarget_SingleModel_allMe
# Import necessary helpers

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




#########
# Fused Model
#########

def h_measure_modelFused(x):
    return x[0:2]

###########

class Track(object):

    def __init__(self, x0, P0):

        self.x = x0
        self.P = P0

class Tracker_SingleTarget_IMultipleModel_allMe(object):
  
    def __init__(self):
                                        
        self.track = None

        self.models = [
            Tracker_SingleTarget_SingleModel_allMe(1), # Constant Velocity(CV) Model 
            Tracker_SingleTarget_SingleModel_allMe(2), # Constant Turn-Rate Velocity(CTRV) Model
            # Tracker_SingleTarget_SingleModel_allMe(3)  # Random Motion(RM) Model
        ]   

        self.modeProbs = np.expand_dims(np.array([
            # 0.34, 0.33, 0.33
                0.5, 0.5
            ]), axis=1)

        self.transitionMatrix = np.array([
            # [0.9, 0.09, 0.01],
            # [0.19, 0.8, 0.01],
            # [0.25, 0.25, 0.5]
            
            [0.95, 0.05],
            [0.05, 0.95]
        ])


        self.measurements = []

        self.Ws, self.Wc, self.lambda_ = uH.generateUnscentedWeights(L = 5, alpha = 1e-3, beta = 2, kappa = 0)

    def fuseStates(self):

        stateMeans = []
        stateCovariances = []

        for model in self.models:
            stateMeans.append(model.track.x)
            stateCovariances.append(model.track.P)

        stateMeans = np.squeeze(np.array(stateMeans, dtype=float))
        stateCovariances = np.array(stateCovariances) 

        fusedX, fusedP = IMM_helper.fuseModelStates(stateMeans, stateCovariances, self.modeProbs)

        return (fusedX, fusedP)

    def mixStates(self):

        stateMeans = []
        stateCovariances = []

        for model in self.models:
            stateMeans.append(model.track.x)
            stateCovariances.append(model.track.P)

        stateMeans = np.squeeze(np.array(stateMeans, dtype=float))
        stateCovariances = np.array(stateCovariances, dtype=float)

        mixedMeans, mixedCovariances = IMM_helper.mixStates(stateMeans, stateCovariances, self.transitionMatrix, self.modeProbs)
        
        for i, model in enumerate(self.models):
            model.x = np.expand_dims(mixedMeans[i], axis=1)
            model.P = mixedCovariances[i]

    def updateModeProbabilities(self, measurement):

        stateMeans_measured = []
        stateSs = []

        stateMeans = []
        stateCovariances = []

        for model in self.models:

            stateMeans_measured.append(model.track.z_predict)
            stateSs.append(model.track.S)   


        stateMeans_measured = np.squeeze(np.array(stateMeans_measured, dtype=float))

        stateSs = np.array(stateSs, dtype=float)
        stateCovariances = np.array(stateCovariances, dtype=float)

        self.modeProbs = IMM_helper.updateModeProbabilities(stateMeans_measured, stateSs, measurement, self.transitionMatrix, self.modeProbs)
            

    def feedMeasurement(self, measurement, dt):

        self.measurements.append(measurement)

        if(self.track is None and len(self.measurements) <= 2):

            #just feed the measurements
            #no track inits or state updates

            for model in self.models:
                model.putMeasurement(measurement)


        elif(self.track is None and len(self.measurements) > 2):

            #just for the init step, mixing won't happen, free run

            for model in self.models:
                model.feedMeasurement(measurement, dt)

            #init track by fusing

            fusedX, fusedP = self.fuseStates()

            self.track = Track(fusedX, fusedP)
            self.track.z = h_measure_modelFused(fusedX)


        elif(self.track is not None):

            #the interaction will now start

            #MIXING

            self.mixStates()
            
            #PREDICTION AND UPDATE

            for model in self.models:
                model.feedMeasurement(measurement, dt)            

            #MODEL PROBS UPDATE

            self.updateModeProbabilities(measurement)

            #FUSE STATES

            #this stage is necessary to update the fused track
            #though there is no use of track.P for now,
            #track.x is obviously our final estimate and important

            fusedX, fusedP = self.fuseStates()
            
            self.track.x = fusedX
            self.track.P = fusedP
            self.track.z = h_measure_modelFused(fusedX)
