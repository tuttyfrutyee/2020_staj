import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation

def getPerimeterPoints(targetPoint, S_inverse, angleStep, gateThreshold):
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
    
    return (x_draw, y_draw)

def showPerimeter(targetPoint, S_inverse, angleStep, gateThreshold):
    """
        Input:
            targetPoint : np.array(shape = (dimX, 1))
            S_inverse : np.array(shape = (dimZ, dimZ))
            angleStep : float
    """

    x_draw, y_draw = getPerimeterPoints(targetPoint, S_inverse, angleStep, gateThreshold)


    plt.scatter(x_draw, y_draw, c = "m", alpha = 0.2, linewidths=1)
    plt.scatter([targetPoint[0][0]], [targetPoint[1][0]], c = "k", linewidths=3)



def showRadius(targetPoint, R, angleStep):
    """
        Input:
            targetPoint : np.array(shape = (dimX, 1))
            R : float
            angleStep : float
    """

    x_draw = []
    y_draw = []

    angle = 0

    numberOfStep = int(np.pi * 2 / angleStep)

    for step in range(numberOfStep):

        angle = step * angleStep

     
        x_draw.append( targetPoint[0][0] + np.cos(angle) * R)
        y_draw.append( targetPoint[1][0] + np.sin(angle) * R)

    plt.scatter(x_draw, y_draw, c = "k")
    plt.scatter([targetPoint[0][0]], [targetPoint[1][0]], c = "b", linewidths=3)
    
    
def visualizeTrackingResult(trackers, measurements, groundTruths, animate, gateThreshold):
    
    predictionPlotColor = ""
    measurementColor = ""
    
    
    if(animate):
        
        fig, ax = plt.subplots()  
        ax.grid()
        
        perimeterX = []
        perimeterY = []
        
        centerX = []
        centerY = []
        
        
        time = 0
        mode = 0 #if mode is 0 push measurement & prediction, if 1 push updated states
        
        drawingTrackers = []
        
        for tracker in trackers:
            drawingTrackers.append([0, [], None]) # historyIndex, historyUntilNow, lastPrediction( [z_prediction, S] )
        
        
        def animation(i):
            
            
            #first clear screen
            ax.clear()
            perimeterX = []
            perimeterY = []
            centerX = []
            centerY = []
            
            #secondly draw the ground truths until now
            for groundTruth in groundTruths:
                scGT = ax.plot(groundTruth[:,0], groundTruth[:,0])
            
            #thirdly draw all trackers until now
            
                #update the drawingTrackers
            for t, drawingTracker in enumerate(drawingTrackers):
                
                correspondingUpdateHistory = trackers[t].updatedStateHistory
                correspondingPredictHistory = trackers[t].predictedStateHistory
                
                if(drawingTracker[0] < len(correspondingUpdateHistory) and correspondingUpdateHistory[drawingTracker[0]][2] == time):
                    
                    x_updated = correspondingUpdateHistory[drawingTracker[0]][0]
                    
                    if(mode == 0):
                        drawingTracker[2] = (correspondingPredictHistory[drawingTracker[0]][0], correspondingPredictHistory[drawingTracker[0]][1])                        
                    elif(mode == 1)
                        drawingTracker[0] += 1
                        drawingTracker[1].append(x_updated)
            
                #now draw all updated states and new predictions
                #also stack the predictionPerimeters
            for drawingTracker in drawingTrackers:
                
                states = np.array(drawingTracker[1])
                scUp = ax.plot(states[:, 0], states[:, 1])
                
                predictionZ, predictionS = drawingTracker[2]
                
                x_perimeter_draw, y_perimeter_draw = getPerimeterPoints(predictionZ, np.linalg.eig(S_inverse), np.pi/180, gateThreshold)
                
                perimeterX += x_perimeter_draw
                perimeterY += y_perimeter_draw
                
                centerX += predictionZ[0]
                centerY += predictionZ[1]
            
            #fourth, draw perimeters and centers
            scPerimeter, = ax.scatter(perimeterX, perimeterY, c = "m")
            scCenter, = ax.scatter(centerX, centerY, c = "k")                
            
            #fifth draw measurements up until now
                
            #finally auto scaling the screen
            ax.relim()
            ax.autoscale_view(True,True,True)
            
            if(mode == 1):
                timeStamp += 1
                
            mode = (mode + 1) % 2
        
        print("todo")
        
    else:
        print("todo")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
