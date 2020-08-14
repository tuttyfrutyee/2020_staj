# -*- coding: utf-8 -*-

import numpy as np
import myHelpers.unscentedHelper as uH

import myHelpers.immHelper as IMM_helper

# Will use single model to create different models
from Trackers.SingleTarget.allMe.track_singleTarget_singleModel import Tracker_SingleTarget_SingleModel_allMe
# Import necessary helpers

measurementNoiseStd = np.sqrt(4)
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
        [0, 0, 0, 0, 0]        

    ],
    #modeltype 2
    [
        [Q_0, 0, 0, 0, 0],
        [0, Q_0, 0, 0, 0],
        [0, 0, Q_0 / 1e2, 0, 0],
        [0, 0, 0, Q_0, 0],
        [0, 0, 0, 0, Q_0 / 1e8]             
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
    ]

]



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
            Tracker_SingleTarget_SingleModel_allMe(3)  # Random Motion(RM) Model
        ]   

        self.modeProbs = np.expand_dims(np.array([
            0.34, 0.33, 0.33
        ]), axis=1)

        self.transitionMatrix = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9]
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

            stateMeans_measured.append(model.track.z)
            stateSs.append(model.track.S)   

            stateMeans.append(model.track.x)
            stateCovariances.append(model.track.P)

        stateMeans_measured = np.squeeze(np.array(stateMeans_measured, dtype=float))
        stateMeans = np.squeeze(np.array(stateMeans, dtype=float))

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
            #though there is no use of track.P for now
            #track.x is obviously our final estimate and important

            fusedX, fusedP = self.fuseStates()
            
            self.track.x = fusedX
            self.track.P = fusedP
            self.track.z = h_measure_modelFused(fusedX)
