import math
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("./Scenarios/objectGenerator")
from pathDraw import drawPath



class Object_(object):
    
    """

        xCornerPoints : [x1, x2, ..., xn] -> the points x values where curvatures are drawn w.r.t
        yCornerPoints : [y1, y2, ..., yn] -> the points y values where curvatures are drawn w.r.t

        xPath : List -> the all path(xvalues) #ground truth values
        yPath : List -> the all path(yvalues) #ground truth values

        xNoisyPath : List -> the all path measured(xvalues) -> note that these values may containt None, if corruption is selected
        yNoisyPath : List -> the all path measured(yvalues) -> note that these values may containt None, if corruption is selected

        corruptedPoints_x : the x value of the corrupted points -> reason for this is to see what would the measurements would look like
        corruptedPoints_y : the y value of the corrupted points -> reason for this is to see what would the measurements would look like


    """

    maxW = 150 # -> generated x points satisfy |x| < maxW
    maxH = 150 # -> generated y points satisfy |y| < maxH


    

    def __init__(self, colors, xCornerPoints = None, yCornerPoints = None, numberOfPoints = None, step_size=0.5):

        """
            If xCornerPoints & yCornerPoints are None than the points(n=numberOfPoints) will be generated randomly
        """
        self.colors = colors # -> colors : (pathColor : 'color', measurementColor : 'color')
        self.step_size = step_size
        
        self.corruptedPoints_x = [] #the points where measurements are None
        self.corruptedPoints_y = [] #the points where measurements are None        

        if(numberOfPoints):
            
            self.xCornerPoints = []
            self.yCornerPoints = []

            for _ in range(numberOfPoints):
                self.xCornerPoints.append(math.floor(np.random.rand() * (self.maxW-1) * 2 - self.maxW)) 
                self.yCornerPoints.append(math.floor(np.random.rand() * (self.maxH-1) * 2 - self.maxH))

        else:

            self.xCornerPoints = xCornerPoints
            self.yCornerPoints = yCornerPoints


        #now draw paths
        self.xPath, self.yPath = drawPath(self.xCornerPoints, self.yCornerPoints, self.step_size)

    def generateNoisyMeasurements(self, std, corruptionCount, corruptionLength):

        self.xNoisyPath = self.xPath.copy()
        self.yNoisyPath = self.yPath.copy()

        #add noise
        self.xNoisyPath += np.random.normal(0, std, (len(self.xNoisyPath)))
        self.xNoisyPath = self.xNoisyPath.tolist()

        self.yNoisyPath += np.random.normal(0, std, (len(self.yNoisyPath)))
        self.yNoisyPath = self.yNoisyPath.tolist()


        #add corruption
        if(corruptionCount and corruptionCount > 0):
            
            choiceBag = list(range(0,len(self.xNoisyPath) - corruptionLength - 1))
            np.random.shuffle(choiceBag)

            for i in range(corruptionCount):

                corruptionStartPoint = choiceBag[i]

                for j in range(corruptionLength):
                    
                    self.corruptedPoints_x.append(self.xPath[corruptionStartPoint + j])
                    self.corruptedPoints_y.append(self.yPath[corruptionStartPoint + j])

                    self.xNoisyPath[corruptionStartPoint + j] = None
                    self.yNoisyPath[corruptionStartPoint + j] = None



    def drawObject(self, figureNumber = None):

        if(figureNumber):
            plt.figure(figureNumber)
        else:
            plt.figure()
        
        plt.title("Object Drawing")

        #first plot path
        plt.plot(self.xPath, self.yPath, self.colors[0], linewidth=2)

        #second plot noisyPath
        #plt.plot(self.xNoisyPath, self.yNoisyPath, c=self.colors[1])

        #third mark the missed measurements
        plt.scatter(self.corruptedPoints_x, self.corruptedPoints_y, c="k", linewidth=3)



