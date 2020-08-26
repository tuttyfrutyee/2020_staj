# -*- coding: utf-8 -*-

import numpy as np

import myHelpers.unscentedHelper as uH
import myHelpers.jpdaHelper as jH
import myHelpers.pdaHelper as pH

import myHelpers.immHelper as IMM_helper

import copy


measurementNoiseStd = np.sqrt(2)
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

def normalizeState(x):

    x[2] = putAngleInRange(x[2])
    x[4] = putAngleInRange(x[4])

    return x



########


MeasurementNoiseCov = \
    [
        [measurementNoiseStd * measurementNoiseStd, 0],
        [0, measurementNoiseStd * measurementNoiseStd]
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
       # [0.12958419304911287,0,0,0,0],
       # [0,0.20416385918814656,0,0,0],
       # [0,0,0.008794949000079913,0,0],
       # [0,0,0,0.8057826337426066,0],
       # [0,0,0,0,0] 
     
        # [Q_0, 0, 0, 0, 0],
        # [0, Q_0, 0, 0, 0],
        # [0, 0, Q_0 / 1e2, 0, 0],
        # [0, 0, 0, Q_0, 0],
        # [0, 0, 0, 0, 0]       
     
       [0,0,0,0,0],
       [0,0,0,0,0],
       [0,0,0,0,0],
       [0,0,0,0,0],
       [0,0,0,0,0]
       
    ]) / 400).tolist(),
    #modeltype 2
    (np.array([
        
       # [0.114736907423371,0,0,0,0],
       # [0,0.1354455356615292,0,0,0],
       # [0,0,0.6637200640035631,0,0],
       # [0,0,0,2.9248106675773875,0],
       # [0,0,0,0,0.9305139758546961]      
     
        # [Q_0, 0, 0, 0, 0],
        # [0, Q_0, 0, 0, 0],
        # [0, 0, Q_0 / 1e2, 0, 0],
        # [0, 0, 0, Q_0, 0],
        # [0, 0, 0, 0, Q_0 / 1e8] 
     
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],     
         
     ])/ 800).tolist(),
    #modeltype 3
    [
        [1e-2, 0, 0, 0, 0],
        [0, 1e-2, 0, 0, 0],
        [0, 0, 0 , 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0 ]             
     ],     
]

MeasurementNoiseCov = np.array(MeasurementNoiseCov, dtype = float)

for i, noiseCov in enumerate(ProcessNoiseCovs):
    ProcessNoiseCovs[i] = np.array(noiseCov, dtype = float)




InitialStartCovs_withoutTimeDivision = [
    
    None,

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


# Model Forward Functions


############
# Model 1
############
def f_predict_model1(x, dt):

        
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
def f_predict_model2(x, dt):

    
    X_new = np.copy(x)
            
    x_new = x[0] + x[3] / x[4] * ( -np.sin(x[2]) + np.sin( x[2] + dt * x[4] ) )
    y_new = x[1] + x[3] / x[4] * ( np.cos(x[2])  - np.cos( x[2] + dt * x[4] ) )
    
    phi_new = x[2] + dt * x[4] 
        
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
        self.P_init = P0
        self.x_predict = None
        self.P_predict = None
        self.z_predict = None
        self.S = None
        self.kalmanGain = None

################
# Tracker Models
################


################
# 1 : (CV) Non-Linear Model, Constant Velocity, Zero Turn Rate
################

class TrackerModel_1(object):

    def __init__(self, initMeasurements, dt, unscentedWeights):

        #init variables

        self.Ws, self.Wc, self.lambda_ = unscentedWeights

        self.trackHistory = []
        self.mixedStateHistory = []


        #find x0 and P0

        dx = (initMeasurements[-1][0] - initMeasurements[-2][0]) / dt
        dy = (initMeasurements[-1][1] - initMeasurements[-2][1]) / dt
        phi = np.arctan(dy / dx)
        vel = np.sqrt(dx**2, dy**2)
        dphi = 0

        x0 = np.array([initMeasurements[-1][0], initMeasurements[-1][1], phi, vel, dphi], dtype=float).reshape((5,1))

        P0 = np.array(InitialStartCovs_withoutTimeDivision[1]) * \
            [[1,1,1,1/dt,1/dt],
            [1,1,1,1/dt,1/dt],
            [1,1,1,1/dt,1/dt],
            [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)],
            [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)]]

        P0 = massageToCovariance(P0, 1e-6)

        self.track = Track(x0, P0)
        self.track.z = h_measure_model1(x0)

    def predict(self, dt):

        self.track.P = massageToCovariance(self.track.P, 1e-8)

        
        sigmaPoints = uH.generateSigmaPoints(self.track.x, self.track.P, self.lambda_)   

        self.track.x_predict, self.track.P_predict = uH.predictNextState(f_predict_model1, dt, sigmaPoints, self.Ws, self.Wc, ProcessNoiseCovs[1])                    

        self.track.x_predict = normalizeState(self.track.x_predict)


        self.track.P_predict = massageToCovariance(self.track.P_predict, 1e-8)
        sigmaPoints = uH.generateSigmaPoints(self.track.x_predict, self.track.P_predict, self.lambda_)                    
        self.track.S, self.track.kalmanGain, self.track.z_predict = uH.calculateUpdateParameters(self.track.x_predict, self.track.P_predict, h_measure_model1, sigmaPoints, self.Ws, self.Wc, MeasurementNoiseCov ) 

    def feedMeasurements(self, measurements, associationProbs, timeStamp):

        x_updated, P_updated, _ = pH.pdaPass(self.track.kalmanGain, associationProbs, measurements, self.track.x_predict, self.track.z_predict, self.track.P_predict, self.track.S)

        x_updated = normalizeState(x_updated)

        self.track.x = x_updated
        self.track.P = P_updated

        self.trackHistory.append([copy.deepcopy(self.track), timeStamp])


################
# 2 : (CTRV) Non-Linear Model, Constant Velocity, Nonzero Turn Rate
################

class TrackerModel_2(object):

    def __init__(self, initMeasurements, dt, unscentedWeights):

        #init variables

        self.Ws, self.Wc, self.lambda_ = unscentedWeights

        self.trackHistory = []
        self.mixedStateHistory = []


        #find x0 and P0        

        dx1 = (initMeasurements[-2][0] - initMeasurements[-3][0]) / dt
        dy1 = (initMeasurements[-2][1] - initMeasurements[-3][1]) / dt     
        
        dx2 = (initMeasurements[-1][0] - initMeasurements[-2][0]) / dt
        dy2 = (initMeasurements[-1][1] - initMeasurements[-2][1]) / dt
        
        phi1 = np.arctan(dy1 / dx1)
        phi2 = np.arctan(dy2 / dx2)
        
        vel = np.sqrt(dx2**2, dy2**2)
        dphi = (phi2 - phi1) / dt

        x0 = np.array([initMeasurements[-1][0], initMeasurements[-1][1], phi2, vel, dphi], dtype=float).reshape((5,1))

        P0 = np.array(InitialStartCovs_withoutTimeDivision[2]) * \
            [[1,1,1,1/dt,1/dt],
            [1,1,1,1/dt,1/dt],
            [1,1,1,1/dt,1/dt],
            [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)],
            [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)]]

        P0 = massageToCovariance(P0, 1e-6)

        self.track = Track(x0, P0)
        self.track.z = h_measure_model2(x0)

    def predict(self, dt):

        self.track.P = massageToCovariance(self.track.P, 1e-8)
        

        sigmaPoints = uH.generateSigmaPoints(self.track.x, self.track.P, self.lambda_)


        self.track.x_predict, self.track.P_predict = uH.predictNextState(f_predict_model2, dt, sigmaPoints, self.Ws, self.Wc, ProcessNoiseCovs[2])

        self.track.x_predict = normalizeState(self.track.x_predict)

        self.track.P_predict = massageToCovariance(self.track.P_predict, 1e-8)               
        sigmaPoints = uH.generateSigmaPoints(self.track.x_predict, self.track.P_predict, self.lambda_)                    
        self.track.S, self.track.kalmanGain, self.track.z_predict = uH.calculateUpdateParameters(self.track.x_predict, self.track.P_predict, h_measure_model2, sigmaPoints, self.Ws, self.Wc, MeasurementNoiseCov ) 
    
    def feedMeasurements(self, measurements, associationProbs, timeStamp):

        x_updated, P_updated, _ = pH.pdaPass(self.track.kalmanGain, associationProbs, measurements, self.track.x_predict, self.track.z_predict, self.track.P_predict, self.track.S)

        x_updated = normalizeState(x_updated)

        self.track.x = x_updated
        self.track.P = P_updated    

        self.trackHistory.append([copy.deepcopy(self.track), timeStamp])


################
# 3 : (RM) Non-Linear Model, Random Motion
################

class TrackerModel_3(object):

    def __init__(self, initMeasurements, dt, unscentedWeights):

        #init variables

        self.Ws, self.Wc, self.lambda_ = unscentedWeights

        self.trackHistory = []
        self.mixedStateHistory = []

        #find x0 and P0        


        x0 = np.array([initMeasurements[-1][0], initMeasurements[-1][1], 0, 0, 0], dtype=float).reshape((5,1))

        P0 = np.array(InitialStartCovs_withoutTimeDivision[3]) * \
            [[1,1,1,1/dt,1/dt],
            [1,1,1,1/dt,1/dt],
            [1,1,1,1/dt,1/dt],
            [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)],
            [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)]]

        P0 = massageToCovariance(P0, 1e-6)

        self.track = Track(x0, P0)
        self.track.z = h_measure_model3(x0)



    def predict(self, dt):

        self.track.P = massageToCovariance(self.track.P, 1e-8)
        
        sigmaPoints = uH.generateSigmaPoints(self.track.x, self.track.P, self.lambda_)

        self.track.x_predict, self.track.P_predict = uH.predictNextState(f_predict_model3, dt, sigmaPoints, self.Ws, self.Wc, ProcessNoiseCovs[3])

        self.track.x_predict = normalizeState(self.track.x_predict)

        self.track.P_predict = massageToCovariance(self.track.P_predict, 1e-8)                    
        sigmaPoints = uH.generateSigmaPoints(self.track.x_predict, self.track.P_predict, self.lambda_)                    
        self.track.S, self.track.kalmanGain, self.track.z_predict = uH.calculateUpdateParameters(self.track.x_predict, self.track.P_predict, h_measure_model3, sigmaPoints, self.Ws, self.Wc, MeasurementNoiseCov ) 

    def feedMeasurements(self, measurements, associationProbs, timeStamp):

        x_updated, P_updated, _ = pH.pdaPass(self.track.kalmanGain, associationProbs, measurements, self.track.x_predict, self.track.z_predict, self.track.P_predict, self.track.S)

        x_updated = normalizeState(x_updated)

        self.track.x = x_updated
        self.track.P = P_updated     

        self.trackHistory.append([copy.deepcopy(self.track), timeStamp])


class MultiModelTracker(object):

    def __init__(self, unscentedWeights, gateThreshold, PD):

        self.unscentedWeights = unscentedWeights

        self.gateThreshold = gateThreshold
        self.PD = PD

        self.modeProbs = None
        self.transitionMatrix = None
     
        self.models = []

        self.track = None

        self.measurements = [] #only for initialization, probably only 3 measurements will be here, then it will be mature


        self.trackerStatus = 0
        self.trackerLifeTime = 0

        self.updatedStateHistory = []
        self.predictedStateHistory = []       
        
    #private functions

    def fuseStates(self):

        stateMeans = []
        stateCovariances = []

        for model in self.models:
            stateMeans.append(model.track.x)
            stateCovariances.append(model.track.P)

        stateMeans = np.squeeze(np.array(stateMeans, dtype=float))
        if(len(self.models) == 1):    
            stateMeans = np.expand_dims(stateMeans, axis=0)
            
        stateCovariances = np.array(stateCovariances, dtype=float) 

        fusedX, fusedP = IMM_helper.fuseModelStates(stateMeans, stateCovariances, self.modeProbs)

        fusedX = normalizeState(fusedX)

        return (fusedX, fusedP)

    def fuseStates_predict(self):

        stateMeans = []
        stateCovariances = []

        for model in self.models:
            stateMeans.append(model.track.x_predict)
            stateCovariances.append(model.track.P_predict)

        stateMeans = np.squeeze(np.array(stateMeans, dtype=float))
        if(len(self.models) == 1):    
            stateMeans = np.expand_dims(stateMeans, axis=0)
            
        stateCovariances = np.array(stateCovariances, dtype=float) 

        fusedX_predict, fusedP_predict = IMM_helper.fuseModelStates(stateMeans, stateCovariances, self.modeProbs)

        fusedX_predict = normalizeState(fusedX_predict)

        return (fusedX_predict, fusedP_predict)


    def mixStates(self, timeStamp):

        stateMeans = []
        stateCovariances = []

        for model in self.models:
            stateMeans.append(model.track.x)
            stateCovariances.append(model.track.P)

        stateMeans = np.squeeze(np.array(stateMeans, dtype=float))
        if(len(self.models) == 1):    
            stateMeans = np.expand_dims(stateMeans, axis=0)
            
        stateCovariances = np.array(stateCovariances, dtype=float)

        mixedMeans, mixedCovariances = IMM_helper.mixStates(stateMeans, stateCovariances, self.transitionMatrix, self.modeProbs)
        
        for i, model in enumerate(self.models):
            beforeMixX = model.track.x
            beforeMixP = model.track.P

            #real deal update
            model.track.x = np.expand_dims(mixedMeans[i], axis=1)
            model.track.P = mixedCovariances[i] 

            #normalize the state
            model.track.x = normalizeState(model.track.x)  

            #update history
            model.mixedStateHistory.append([beforeMixX, beforeMixP, model.track.x, model.track.P, timeStamp])


    def updateModeProbabilities(self, measurements):

        modeStateMeans_measured = []
        modeSs = []

        for model in self.models:

            modeStateMeans_measured.append(model.track.z_predict)
            modeSs.append(model.track.S)

        modeStateMeans_measured = np.array(modeStateMeans_measured, dtype = float)
        modeSs = np.array(modeSs, dtype = float)

        if(measurements.shape[0] != 0):
            updatedModeProbabilities = IMM_helper.updateModeProbabilities_PDA(modeStateMeans_measured, modeSs, measurements, self.gateThreshold, self.PD, self.transitionMatrix, self.modeProbs)           
        else:    
            updatedModeProbabilities = self.modeProbs

        self.modeProbs = updatedModeProbabilities

    #public functions

    def putMeasurement(self, measurement, dt):

        self.trackerLifeTime += 1

        if(measurement is None):
            self.trackerStatus = 0

        else:
            self.measurements.append(measurement)
            self.trackerStatus += 1

            if(self.trackerStatus == 3):

                model1 = TrackerModel_1(self.measurements, dt, self.unscentedWeights)
                model2 = TrackerModel_2(self.measurements, dt, self.unscentedWeights)
                # model3 = TrackerModel_3(self.measurements, dt, self.unscentedWeights)

                self.models.append(model1)
                self.models.append(model2)
                # self.models.append(model3)  

                self.modeProbs = np.expand_dims(np.array([
                    # 0.5, 0.49, 0.01
                    0.5, 0.5
                    # 1
                ]), axis=1)

                self.transitionMatrix = np.array([
                    # [0.9, 0.05, 0.05],
                    # [0.05, 0.9, 0.05],
                    # [0.2, 0.2, 0.6]
                    
                    [0.9, 0.1],
                    [0.1, 0.9]
                    
                    # [1]
                ])                

                fusedX, fusedP = self.fuseStates()

                self.track = Track(fusedX, fusedP)     

    def predict(self, dt, timeStamp):

        self.trackerLifeTime += 1

        self.mixStates(timeStamp)

        for model in self.models:
            model.predict(dt)  
            
        # self.track = copy.deepcopy(self.models[0].track)
        

        fusedX_predict, fusedP_predict = self.fuseStates_predict()

        self.track.x_predict = fusedX_predict
        self.track.P_predict = fusedP_predict
        #todo, maybe explain here?
        self.track.z_predict = fusedX_predict[0:2]
        self.track.S = self.track.P_predict[0:2, 0:2] + MeasurementNoiseCov

        self.predictedStateHistory.append((self.track.z_predict, self.track.S, timeStamp))


    def feedMeasurements(self, measurements, associationProbs, timeStamp):
        
        if(associationProbs is None or np.sum(associationProbs) == 0):
            self.trackerStatus -= 1
        else:
            self.trackerStatus = 8

        for model in self.models:
            model.feedMeasurements(measurements, associationProbs, timeStamp)            

        self.updateModeProbabilities(measurements)

        fusedX, fusedP = self.fuseStates()

        self.track.x = fusedX
        self.track.P = fusedP

        self.updatedStateHistory.append((fusedX, fusedP, timeStamp))


class Tracker_MultipleTarget_MultipleModel_allMe(object):
  
    def __init__(self, gateThreshold, distanceThreshold, detThreshold, spatialDensity, PD):
                        
        self.matureTrackers = []
        self.initTrackers = []

        self.matureTrackerHistory = []

        self.unscentedWeights = None
        self.validationMatrix = None

        self.gateThreshold = gateThreshold
        self.distanceThreshold = distanceThreshold
        self.spatialDensity = spatialDensity
        self.detThreshold = detThreshold
        self.PD = PD


        self.unscentedWeights = uH.generateUnscentedWeights(L = 5, alpha = 1e-3, beta = 2, kappa = 0)

    #private functions

    def initNewTrackers(self, measurements, dt):

        for measurement in measurements:

            newTracker = MultiModelTracker(self.unscentedWeights, self.gateThreshold, self.PD)
            newTracker.putMeasurement(measurement, dt)

            self.initTrackers.append(newTracker)

    def deleteDeadTrackers(self):
        
        toDelete_matureTrackers = []
        toDelete_initTrackers = []
        
        for tracker in self.matureTrackers:
            if(tracker.trackerStatus == 0):
                toDelete_matureTrackers.append(tracker)
            
            elif( tracker.trackerLifeTime > 10 and np.linalg.det(tracker.track.P) > self.detThreshold):
                toDelete_matureTrackers.append(tracker)
                print("BOOM DEAD")
            
        for tracker in self.initTrackers:
            if(tracker.trackerStatus == 0):
                toDelete_initTrackers.append(tracker)
        
        for delTracker in toDelete_matureTrackers:
            self.matureTrackers.remove(delTracker)
        for delTracker in toDelete_initTrackers:
            self.initTrackers.remove(delTracker)

    def trackertify(self):

        newMatureTracks = []

        for tracker in self.initTrackers:
            if(tracker.trackerStatus > 2):
                newMatureTracks.append(tracker)

        for tracker in newMatureTracks:
            self.initTrackers.remove(tracker)
            self.matureTrackers.append(tracker)
            self.matureTrackerHistory.append(tracker)
            
            

    def predict(self, dt, timeStamp):

        for tracker in self.matureTrackers:
            tracker.predict(dt, timeStamp)

    #public functions



    def feedMeasurements(self, measurements, dt, timeStamp):

        #first delete dead trackers
        self.deleteDeadTrackers()

        #second select from initTracks that are mature now
        self.trackertify()

        #now predict the next state, only mature trackers will predict
        self.predict(dt, timeStamp)

        #greedy association to find unmatched measurements
        matureTrackers = np.array(self.matureTrackers, dtype=object)
        initTrackers = np.array(self.initTrackers, dtype = object)
        unmatchedMeasurements, initTrackerBoundedMeasurements, distanceMatrix = jH.greedyAssociateMeasurements(matureTrackers, initTrackers, measurements, self.gateThreshold, self.distanceThreshold)


        #put measurements to init tracks
        for i,tracker in enumerate(self.initTrackers):
            tracker.putMeasurement(initTrackerBoundedMeasurements[i], dt)

        if(len(self.matureTrackers) > 0):


            #now the association probabilities will be calculated(JPDA)

                #createValidationMatrix
            validatedMeasurementIndexes, validationMatrix = jH.createValidationMatrix(distanceMatrix, measurements, self.matureTrackers, self.gateThreshold)
            self.validationMatrix = validationMatrix

            validatedMeasurements = measurements[validatedMeasurementIndexes]
            self.validatedMeasurements = validatedMeasurements

                #generateAssociationEvents
            associationEvents = jH.generateAssociationEvents(validationMatrix)
            self.associationEvents = associationEvents

                #calculateMarginalAssociationProbs
            marginalAssociationProbabilities = jH.calculateMarginalAssociationProbabilities(associationEvents, validatedMeasurements, self.matureTrackers, self.spatialDensity, self.PD)

            self.marginalAssociationProbabilities = marginalAssociationProbabilities

            #now pass the marginalAssocationProbs to PDA stage
            for t,tracker in enumerate(self.matureTrackers):
                
                if(marginalAssociationProbabilities is not None):
                    associationProbs = (marginalAssociationProbabilities.T)[t]
                else:
                    associationProbs = None

                tracker.feedMeasurements(validatedMeasurements, associationProbs, timeStamp)
                
        #finally go and init new tracks with unmatchedMeasurements

        self.initNewTrackers(unmatchedMeasurements, dt)
                