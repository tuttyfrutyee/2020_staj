# -*- coding: utf-8 -*-

import numpy as np

dimX = 5 #state dimension
dimZ = 3 #measurement state dimension
Nr = 3 #number of modes(models)
m_k = 4 #number of measurements


def generateRandomCovariance_positiveDefinite(dim):
    randomCovariance = np.random.randn(dim, dim)
    randomCovariance = 0.5*(randomCovariance + randomCovariance.T) / ( np.max(abs(randomCovariance))) + dim * np.eye(dim)
    return randomCovariance

def generateRandomCovariances_positiveDefinite(n, dim):
    covariances = []
    for i in range(n):
        covariances.append(generateRandomCovariance_positiveDefinite(dim))
    return np.array(covariances, dtype="float")

#IMM
stateMeans = np.random.randn(Nr, dimX)

stateCovariances = generateRandomCovariances_positiveDefinite(Nr, dimX)

transitionMatrix = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])

modeProbs = np.array([0.1, 0.5, 0.4])
modeProbs = np.expand_dims(modeProbs, axis=1)

modePzzs = generateRandomCovariances_positiveDefinite(Nr, dimZ)

measurements = np.random.randn(m_k, dimZ)

likelihoods = abs(np.random.randn(Nr, m_k))
#likelihoods = likelihoods / (np.expand_dims(np.sum(likelihoods, axis=1), axis=1))

gateThreshold = 0.7

PD = 0.9

previousModeProbabilities = abs(np.random.randn(Nr, 1))
previousModeProbabilities = previousModeProbabilities / np.sum(previousModeProbabilities)

#JPDA
meas1 = [1,0,1]
meas2 = [1,1,0]
meas3 = [1,1,1]
meas4 = [1,0,1]

measurements_ = [meas1, meas2, meas3, meas4]

validationMatrix = []
for meas in measurements_:
    validationMatrix.append(np.array(meas))
validationMatrix = np.array(validationMatrix)

class Track(object):
  def __init__(self, z_prior, S_prior):
    self.z_prior = z_prior
    self.S_prior = S_prior

tracks = []
for i in range(Nr):
    z_prior = np.random.randn(dimZ, 1)
    S_prior = generateRandomCovariance_positiveDefinite(dimZ)
    tracks.append(Track(z_prior, S_prior))