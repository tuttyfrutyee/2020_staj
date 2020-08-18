# -*- coding: utf-8 -*-

import numpy as np

import myHelpers.unscentedHelper as uH
import myHelpers.jpdaHelper as jH
import myHelpers.pdaHelper as pH


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
    #modeltype 3
    [
        [measurementNoiseStd * measurementNoiseStd, 0],
        [0, measurementNoiseStd * measurementNoiseStd]
    ],    
]
ProcessNoiseCovs = [
    #modeltype 0
# [[ 0.0251,  0.0219, -0.0072, -0.0054],
#         [ 0.0219,  0.0199, -0.0064, -0.0049],
#         [-0.0072, -0.0064,  0.0023,  0.0016],
#         [-0.0054, -0.0049,  0.0016,  0.0017]],
    [
        [Q_0, 0, 0, 0],
        [0, Q_0, 0, 0],
        [0, 0, Q_0 , 0],
        [0, 0, 0, Q_0],     

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
    #modeltype 3
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1e-2 , 0, 0],
        [0, 0, 0, 4e-1, 0],
        [0, 0, 0, 0, 1e-15 ]             
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


#############
# Model 0
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

############
# Model 1
############
def f_predict_model1(x_, dt):

    x = np.copy(x_)
    
    x[2] = putAngleInRange(x[2])
    
    X_new = np.copy(x)

    x_new = x[0] + x[3] * dt * np.sin(x[2])
    y_new = x[1] + x[3] * dt * np.cos(x[2])
    
    X_new[0] = x_new
    X_new[1] = y_new

    return X_new

def h_measure_model1(x):
    return x[0:2]

############
# Model 2
############
def f_predict_model2(x_, dt):

    
    x = np.copy(x_)
    
    x[2] = putAngleInRange(x[2])
    x[4] = putAngleInRange(x[4])
    
    X_new = np.copy(x)
            
    x_new = x[0] + x[3] / x[4] * ( -np.sin(x[2]) + np.sin( x[2] + dt * x[4] ) )
    y_new = x[1] + x[3] / x[4] * ( np.cos(x[2])  - np.cos( x[2] + dt * x[4] ) )
    
    phi_new = x[2] + dt * x[4] 
    

    phi_new = putAngleInRange(phi_new)
    
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
        self.x_predict = None
        self.P_predict = None
        self.z_predict = None
        self.S = None
        self.kalmanGain = None
        #optional
        self.H = None #only for linear kalman filter


class Tracker(object):

    def __init__(self, x0, P0, modelType, unscentedWeights):
        self.x = x0
        self.P = P0
        self.modelType = modelType

        if(modelType == 1 or modelType == 2 or modelType == 3):
            self.Ws, self.Wc, self.lambda_ = unscentedWeights

        self.measurements = []

        self.track = None

        self.trackerStatus = 0

        self.x_predict = None
        self.P_predict = None
        self.z_predict = None
        self.S = None
        self.kalmanGain = None
        #optional
        self.H = None #only for linear kalman filter
    
    def putMeasurement(self, measurement, dt):
        #can be used to put only measurement to self.measurements without state update
          

        if(measurement is None):
            self.trackerStatus = 0
        else:
            self.measurements.append(measurement)            
            self.trackerStatus += 1

        if(self.trackerStatus > 2):
            #init the track
            if(self.modelType == 0): #linearModel, constant velocity, constant direction
                dx = (self.measurements[-1][0] - self.measurements[-2][0]) / dt
                dy = (self.measurements[-1][1] - self.measurements[-2][1]) / dt
                x0 = np.array([self.measurements[-1][0], self.measurements[-1][1] , dx, dy]).reshape((4,1))

                P0 = np.array(InitialStartCovs_withoutTimeDivision[0]) * \
                    [[1,1,1/dt,1],
                    [1,1,1,1/dt],
                    [1/dt,1,1/(dt**2),1],
                    [1,1/dt,1,1/(dt**2)]]
                    
                P0 = massageToCovariance(P0, 1e-6)

                self.track = Track(x0, P0)
                self.track.z, _ = h_measure_model0(self.track.x)
                
            elif(self.modelType == 1): #nonLinearModel, Constant Velocity (CV) Model   
            
                dx = (self.measurements[-1][0] - self.measurements[-2][0]) / dt
                dy = (self.measurements[-1][1] - self.measurements[-2][1]) / dt
                phi = np.arctan(dy / dx)
                vel = np.sqrt(dx**2, dy**2)
                dphi = 0

                x0 = np.array([self.measurements[-1][0], self.measurements[-1][1], phi, vel, dphi]).reshape((5,1))

                P0 = np.array(InitialStartCovs_withoutTimeDivision[1]) * \
                    [[1,1,1,1/dt,1/dt],
                    [1,1,1,1/dt,1/dt],
                    [1,1,1,1/dt,1/dt],
                    [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)],
                    [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)]]

                P0 = massageToCovariance(P0, 1e-6)

                self.track = Track(x0, P0)
                self.track.z = h_measure_model1(self.track.x)                
                
            elif(self.modelType == 2): #nonLinearModel, Constant Turn-Rate Velocity (CTRV) Model
                if(len(self.measurements) > 2):
                    
                    dx1 = (self.measurements[-2][0] - self.measurements[-3][0]) / dt
                    dy1 = (self.measurements[-2][1] - self.measurements[-3][1]) / dt     
                    
                    dx2 = (self.measurements[-1][0] - self.measurements[-2][0]) / dt
                    dy2 = (self.measurements[-1][1] - self.measurements[-2][1]) / dt
                    
                    phi1 = np.arctan(dy1 / dx1)
                    phi2 = np.arctan(dy2 / dx2)
                    
                    vel = np.sqrt(dx2**2, dy2**2)
                    dphi = phi2 - phi1

                    x0 = np.array([self.measurements[-1][0], self.measurements[-1][1], phi2, vel, dphi]).reshape((5,1))

                    P0 = np.array(InitialStartCovs_withoutTimeDivision[2]) * \
                        [[1,1,1,1/dt,1/dt],
                        [1,1,1,1/dt,1/dt],
                        [1,1,1,1/dt,1/dt],
                        [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)],
                        [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)]]

                    P0 = massageToCovariance(P0, 1e-6)


                    self.track = Track(x0, P0)
                    self.track.z = h_measure_model2(self.track.x)

            elif(self.modelType == 3): # Random Motion (RM) Model 

                if(len(self.measurements) > 2):
                    
                    dx1 = (self.measurements[-2][0] - self.measurements[-3][0]) / dt
                    dy1 = (self.measurements[-2][1] - self.measurements[-3][1]) / dt     
                    
                    dx2 = (self.measurements[-1][0] - self.measurements[-2][0]) / dt
                    dy2 = (self.measurements[-1][1] - self.measurements[-2][1]) / dt
                    
                    phi1 = np.arctan(dy1 / dx1)
                    phi2 = np.arctan(dy2 / dx2)
                    
                    vel = np.sqrt(dx2**2, dy2**2)
                    dphi = phi2 - phi1

                    x0 = np.array([self.measurements[-1][0], self.measurements[-1][1], phi2, vel, dphi]).reshape((5,1))

                    P0 = np.array(InitialStartCovs_withoutTimeDivision[3]) * \
                        [[1,1,1,1/dt,1/dt],
                        [1,1,1,1/dt,1/dt],
                        [1,1,1,1/dt,1/dt],
                        [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)],
                        [1/dt,1/dt,1/dt,1/(dt**2),1/(dt**2)]]

                    P0 = massageToCovariance(P0, 1e-6)


                    self.track = Track(x0, P0)
                    self.track.z = h_measure_model3(self.track.x)              

    def feedMeasurements(self, measurements, associationProbs):
        print("todo")

    def predict(self, dt):

        if(self.track):

            if(self.modelType == 0):

                self.track.x_predict, self.track.P_predict = f_predict_model0(self.track.x, self.track.P, dt)
                self.track.z_predict, self.track.H = h_measure_model0(self.track.x_predict)
                self.track.S = np.dot(self.track.H, np.dot(self.track.P_predict, self.track.H.T)) + MeasurementNoiseCovs[0]
                self.track.kalmanGain = np.dot(self.track.P_predict, np.dot(self.track.H.T, np.linalg.inv(self.track.S)))                


            elif(self.modelType == 1 or self.modelType == 2 or self.modelType == 3):

                self.track.P = massageToCovariance(self.track.P, 1e-8)

                sigmaPoints = uH.generateSigmaPoints(self.track.x, self.track.P, self.lambda_)

                if(self.modelType == 1):
                    
                    self.track.x_predict, self.track.P_predict = uH.predictNextState(f_predict_model1, dt, sigmaPoints, self.Ws, self.Wc, ProcessNoiseCovs[1])                    
                    sigmaPoints = uH.generateSigmaPoints(self.track.x_predict, self.track.P_predict, self.lambda_)                    
                    self.track.S, self.track.kalmanGain, self.track.z_predict = uH.calculateUpdateParameters(self.track.x_predict, self.track.P_predict, h_measure_model1, sigmaPoints, self.Ws, self.Wc, MeasurementNoiseCovs[1] ) 


                elif(self.modelType == 2):

                    self.track.x_predict, self.track.P_predict = uH.predictNextState(f_predict_model2, dt, sigmaPoints, self.Ws, self.Wc, ProcessNoiseCovs[2])
                    sigmaPoints = uH.generateSigmaPoints(self.track.x_predict, self.track.P_predict, self.lambda_)                    
                    self.track.S, self.track.kalmanGain, self.track.z_predict = uH.calculateUpdateParameters(self.track.x_predict, self.track.P_predict, h_measure_model2, sigmaPoints, self.Ws, self.Wc, MeasurementNoiseCovs[2] ) 
                
                elif(self.modelType == 3):
                    
                    self.track.x_predict, self.track.P_predict = uH.predictNextState(f_predict_model3, dt, sigmaPoints, self.Ws, self.Wc, ProcessNoiseCovs[3])
                    sigmaPoints = uH.generateSigmaPoints(self.track.x_predict, self.track.P_predict, self.lambda_)                    
                    self.track.S, self.track.kalmanGain, self.track.z_predict = uH.calculateUpdateParameters(self.track.x_predict, self.track.P_predict, h_measure_model3, sigmaPoints, self.Ws, self.Wc, MeasurementNoiseCovs[3] ) 



class Tracker_MultipleTarget_SingleModel_allMe(object):
  
    def __init__(self, modelType, gateThreshold, distanceThreshold, spatialDensity, PD):
                        
        self.matureTrackers = []
        self.initTrackers = []

        self.unscentedWeights = None
        self.validationMatrix = None

        self.modelType = modelType
        self.gateThreshold = gateThreshold
        self.distanceThreshold = distanceThreshold
        self.spatialDensity = spatialDensity
        self.PD = PD

        if(modelType == 1 or modelType == 2 or modelType == 3):
            self.Ws, self.Wc, self.lambda_ = uH.generateUnscentedWeights(L = 5, alpha = 1e-3, beta = 2, kappa = 0)

    
    def predict(self, dt):

        for tracker in self.matureTrackers:
            tracker.predict(dt)

    def initNewTrackers(self, measurements, dt):

        for measurement in measurements:
            if(self.modelType == 1 or self.modelType == 2 or self.modelType == 3):
                newTracker = Tracker(None, None, self.modelType, (self.Wc, self.Wc, self.lambda_))
            else:
                newTracker = Tracker(None, None, self.modelType, None)
            
            newTracker.putMeasurement(measurement, dt)

            self.initTrackers.append(newTracker)

    def deleteDeadTrackers(self):
        
        toDelete_matureTrackers = []
        toDelete_initTrackers = []
        
        for tracker in self.matureTrackers:
            if(tracker.trackerStatus == 0):
                toDelete_matureTrackers.append(tracker)
            
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
            
    def feedMeasurements(self, measurements, dt):

        #first delete dead trackers
        self.deleteDeadTrackers()

        #second select from initTracks that are mature now
        self.trackertify()

        #now predict the next state, only mature trackers will predict
        self.predict(dt)

        #greedy association to find unmatched measurements
        matureTrackers = np.array(self.matureTrackers, dtype=object)
        initTrackers = np.array(self.initTrackers, dtype = object)
        unmatchedMeasurements, initTrackerBoundedMeasurements, distanceMatrix = jH.greedyAssociateMeasurements(matureTrackers, initTrackers, measurements, self.gateThreshold, self.distanceThreshold)

        """         print("measurement shape ", measurements.shape[0])
        print("len(unmatchedMeasurements)", len(unmatchedMeasurements))
        print("len(initTrackerBoundedMeasurements)", initTrackerBoundedMeasurements)
        print("len(matureTrackers)", len(self.matureTrackers))
        print("len(initTrackers)", len(self.initTrackers))
        print("\n\n\n") """
        #put measurements to init tracks
        for i,tracker in enumerate(self.initTrackers):
            tracker.putMeasurement(initTrackerBoundedMeasurements[i], dt)

        if(len(self.matureTrackers) > 0):

            #now the association probabilities will be calculated(JPDA)

                #createValidationMatrix
            validatedMeasurementIndexes, validationMatrix = jH.createValidationMatrix(distanceMatrix, measurements, self.matureTrackers, self.gateThreshold)
            self.validationMatrix = validationMatrix
            validatedMeasurements = measurements[validatedMeasurementIndexes]

                #generateAssociationEvents
            associationEvents = jH.generateAssociationEvents(validationMatrix)

                #calculateMarginalAssociationProbs
            marginalAssociationProbabilities = jH.calculateMarginalAssociationProbabilities(associationEvents, validatedMeasurements, self.matureTrackers, self.spatialDensity, self.PD)


            #now pass the marginalAssocationProbs to PDA stage
            for t,tracker in enumerate(self.matureTrackers):
                
                associationProbs = (marginalAssociationProbabilities.T)[t]
                
                x_updated, P_updated = pH.pdaPass(tracker.track.kalmanGain, associationProbs, validatedMeasurements, tracker.track.x_predict, tracker.track.z_predict, tracker.track.P_predict, tracker.track.S)

                tracker.track.x = x_updated
                tracker.track.P = P_updated

        #finally go and init new tracks with unmatchedMeasurements

        self.initNewTrackers(unmatchedMeasurements, dt)
                
                