# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("../")
import commonVariables as commonVar


def showPerimeter(targetPoint, S_inverse, angleStep, gateThreshold):
    """
        Input:
            targetPoint : np.array(shape = (dimX, 1))
            S_inverse : np.array(shape = (dimZ, dimZ))
            angleStep : float
    """

    x_draw = []
    y_draw = []

    angle = 0

    numberOfStep = int(np.pi * 2 / angleStep)

    for step in range(numberOfStep):

        angle = step * angleStep

        angleVector = np.array([np.cos(angle), np.sin(angle)]).reshape((2,1))

        

        R = np.sqrt(gateThreshold / np.dot(angleVector.T, np.dot(S_inverse, angleVector)))

        x_draw.append( targetPoint[0][0] + np.cos(angle) * R)
        y_draw.append( targetPoint[1][0] + np.sin(angle) * R)

    plt.scatter(x_draw, y_draw, c = "m")


targetX = np.array([0,0]).reshape((2,1))

showPerimeter(targetX, commonVar.S, np.pi / 100, 1)