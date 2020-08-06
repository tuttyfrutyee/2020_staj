import math
import numpy as np
import matplotlib.pyplot as plt
from pathDraw import drawPath



class Object(object):

    maxW = 200 # -> generated x points satisfy |x| < maxW
    maxH = 200 # -> generated y points satisfy |y| < maxH


    

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
        for i,_ in enumerate(self.xNoisyPath):
            self.xNoisyPath[i] += np.random.normal(0, std) 
            self.yNoisyPath[i] += np.random.normal(0, std)  

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
        plt.scatter(self.xNoisyPath, self.yNoisyPath, c=self.colors[1], linewidth=0.1)

        #third mark the missed measurements
        plt.scatter(self.corruptedPoints_x, self.corruptedPoints_y, c="k", linewidth=3)



