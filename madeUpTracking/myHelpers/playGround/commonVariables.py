# -*- coding: utf-8 -*-

import numpy as np

dimX = 5 #state dimension
dimZ = 2 #measurement state dimension
Nr = 3 #number of modes(models)
m_k = 3 #number of measurements


def generateRandomCovariance_positiveDefinite(dim):
    randomCovariance = np.random.randn(dim, dim)
    randomCovariance = 0.5*(randomCovariance + randomCovariance.T) / ( np.max(abs(randomCovariance))) + dim * np.eye(dim)
    return randomCovariance

def generateRandomCovariances_positiveDefinite(n, dim):
    covariances = []
    for i in range(n):
        covariances.append(generateRandomCovariance_positiveDefinite(dim))
    return np.array(covariances, dtype="float")

class Track_(object):

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

    def __init__(self, modelType, unscentedWeights):

        self.modelType = modelType

        if(modelType == 1 or modelType == 2 or modelType == 3):
            self.Ws, self.Wc, self.lambda_ = unscentedWeights

        self.measurements = []

        self.track = None

        self.updatedStateHistory = []
        self.predictedStateHistory = []

        self.trackerStatus = 0

        self.x_predict = None
        self.P_predict = None
        self.z_predict = None
        self.S = None
        self.kalmanGain = None
        #optional
        self.H = None #only for linear kalman filter

#IMM
stateMeans = np.random.randn(Nr, dimX)
stateMean = np.random.randn(dimX,1)

stateMeans_measured = np.random.randn(Nr, dimZ)

stateCovariances = generateRandomCovariances_positiveDefinite(Nr, dimX)
stateCovariances_measured = generateRandomCovariances_positiveDefinite(Nr, dimZ)

transitionMatrix = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])

modeProbs = np.array([0.1, 0.5, 0.4])
modeProbs = np.expand_dims(modeProbs, axis=1)

modeSs = generateRandomCovariances_positiveDefinite(Nr, dimZ)

measurements = np.random.randn(m_k, dimZ)

likelihoods = abs(np.random.randn(Nr, m_k))
#likelihoods = likelihoods / (np.expand_dims(np.sum(likelihoods, axis=1), axis=1))

gateThreshold = 0.8

PD = 0.9

previousModeProbabilities = abs(np.random.randn(Nr, 1))
previousModeProbabilities = previousModeProbabilities / np.sum(previousModeProbabilities)

#JPDA

class Track(object):
  def __init__(self, z_prior, S):
    self.z_prior = z_prior
    self.S = S

tracks = []
maxDet = 0
for i in range(Nr):
    z_prior = np.random.randn(dimZ, 1)
    S = generateRandomCovariance_positiveDefinite(dimZ)
    det = np.linalg.det(S)
    if(det > maxDet):
        maxDet = det
    tracks.append(Track(z_prior, S))
    

spatialDensity = 0.1
distanceThreshold = 1

matureTrackers = []
initTrackers = []

for i in range(3):
    tracker = Tracker(0, None)
    tracker.track = Track_(np.random.rand(dimX, 1), generateRandomCovariance_positiveDefinite(dimX))
    tracker.track.z_predict = tracker.track.x[0:dimZ].reshape(dimZ,1)
    tracker.track.S = generateRandomCovariance_positiveDefinite(dimZ)  
    
    matureTrackers.append(tracker)
    
for i in range(2):
    tracker = Tracker(0, None)
    tracker.track = Track_(np.random.rand(dimX, 1), generateRandomCovariance_positiveDefinite(dimX))
    tracker.track.z_predict = tracker.track.x[0:dimZ].reshape(dimZ,1)
    tracker.track.S = generateRandomCovariance_positiveDefinite(dimZ)    
    tracker.measurements = [np.random.randn(dimZ, 1)]
    
    initTrackers.append(tracker)

matureTrackers = np.array(matureTrackers, dtype=object)
initTrackers = np.array(initTrackers, dtype=object)
    
#PDA

kalmanGain = np.random.randn(dimX, dimZ)
priorStateMean = np.random.randn(dimX,1)
priorStateMeasuredMean = np.random.randn(dimZ,1)
priorStateCovariance = np.random.randn(dimX, dimX)
S = generateRandomCovariance_positiveDefinite(dimZ)

#UKF

L = dimX
alpha = 0.01
beta = 2    
kappa = 0

dt = 0.1

processNoise = generateRandomCovariance_positiveDefinite(dimX)
measurementNoise = generateRandomCovariance_positiveDefinite(dimZ)










