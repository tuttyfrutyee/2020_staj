# -*- coding: utf-8 -*-
import numpy as np
import commonVariables as commonVar

print_common = False

def print_(*element):
    if(print_common):
        print(element)



def calculateNDimensionalUnitVolume(n): #checkCount : 1, confident
    return np.power(np.pi, n/2) / gamaFunc(n/2 + 1)

def gamaFunc(n): #checkCount : 1, confident

    gama_1over2 = np.sqrt(np.pi)
    runningProduct = 1
    
    while(n > 1):
        runningProduct *= (n-1)
        n = n - 1
    if(n == 0.5):
        runningProduct *= gama_1over2
    return runningProduct


def calculateGatedVolume(nz, gateThreshold, detS): #checkCount : 1
    return calculateNDimensionalUnitVolume(nz) * np.power(gateThreshold, nz/2) * np.sqrt(detS) #[BYL95 page 130 3.4.1-6]




#playground

print_(calculateNDimensionalUnitVolume(1), "should be = ",2)
print_(calculateNDimensionalUnitVolume(2), "should be = ",np.pi)
print_(calculateNDimensionalUnitVolume(3), "should be = ", np.pi * 4 / 3)

gatedVolume = calculateGatedVolume(commonVar.dimZ, commonVar.gateThreshold, commonVar.maxDet)
print_("gated volume : ", gatedVolume)