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
    
    
def visualizeTrackingResults(trackers, measurementPacks, groundTruthPacks, animate, gateThreshold):
    
    predictionPlotColor = ""
    measurementColor = ""
    
    
    if(animate):
        
        fig, ax = plt.subplots()  
        ax.grid()
           
            
        drawingTrackers = []
        
        for tracker in trackers:
            drawingTrackers.append([0, [], None]) # historyIndex, historyUntilNow, lastPrediction( [z_prediction, S] )
        
        
        def animation(i):
            
            
            if not hasattr(animation, "time"):        
                 animation.time = 0    
            if not hasattr(animation, "mode"):        
                 animation.mode = 0    

            if not hasattr(animation, "perimeterX"):        
                 animation.perimeterX = []    
            if not hasattr(animation, "perimeterY"):        
                 animation.perimeterY = []    

            if not hasattr(animation, "centerXs"):        
                 animation.centerXs = []    
            if not hasattr(animation, "centerYs"):        
                 animation.centerYs = []             

            if not hasattr(animation, "measurementX"):        
                 animation.measurementX = []    
            if not hasattr(animation, "measurementY"):        
                 animation.measurementY = []   



            time = animation.time
            mode = animation.mode  

            perimeterX = animation.perimeterX
            perimeterY = animation.perimeterY
            
            centerXs = animation.centerXs
            centerYs = animation.centerYs
            
            measurementX = animation.measurementX
            measurementY = animation.measurementY

            print(time)
              
            
            #print(time)
            #first clear screen
            ax.clear()

            if(mode == 0):

                centerXs = []
                centerYs = []

                measurementX = []
                measurementY = []

                perimeterX = []
                perimeterY = []
                      
                       
            #secondly draw the ground truths until now
            for groundTruthPack in groundTruthPacks:
                ax.plot(groundTruthPack[0], groundTruthPack[1])
            
            #thirdly draw all trackers until now
            
                #update the drawingTrackers
            for t, drawingTracker in enumerate(drawingTrackers):
                
                correspondingUpdateHistory = trackers[t].updatedStateHistory
                correspondingPredictHistory = trackers[t].predictedStateHistory
                
                
                if(drawingTracker[0] < len(correspondingUpdateHistory) and correspondingUpdateHistory[drawingTracker[0]][2] == time):
                    
                    x_updated = correspondingUpdateHistory[drawingTracker[0]][0]
                    
                    if(mode == 0):
                        drawingTracker[2] = (correspondingPredictHistory[drawingTracker[0]][0], correspondingPredictHistory[drawingTracker[0]][1])                        
                    elif(mode == 1):
                        drawingTracker[0] += 1
                        drawingTracker[1].append(x_updated)
            
                #now draw all updated states and new predictions
                #also stack the predictionPerimeters
            for drawingTracker in drawingTrackers:
                
                if(len(drawingTracker[1]) > 0):

                    states = np.array(drawingTracker[1])
                
                    ax.plot(states[:, 0], states[:, 1])
                    
                    if(mode == 0):
                        # print("##############################\n \n")
                        # print("1")
                        predictionZ, predictionS = drawingTracker[2]
                        # print("2")
                        
                        x_perimeter_draw, y_perimeter_draw = getPerimeterPoints(predictionZ, np.linalg.inv(predictionS), np.pi/180, gateThreshold)
                        # print("3")
                        
                        perimeterX += x_perimeter_draw
                        perimeterY += y_perimeter_draw
                        # print("4")

                        centerX = []
                        centerY = []
                        # print("5")

                        centerX.append(states[-1, 0][0]) 
                        centerY.append(states[-1, 1][0]) 
                        # print("6")

                        centerX.append(predictionZ[0][0]) 
                        centerY.append(predictionZ[1][0]) 
                        # print("7")

                        # print(centerX)
                        # print(centerY)

                        centerXs.append(centerX)
                        centerYs.append(centerY)

                        # print("8\n\n")




            # print("before update measurements")
            #update measurements if mode is 0
            if(mode == 0): #update the measurements
            
                measurementX = []
                measurementY = []

                measurementPack = measurementPacks[time]
                
                for measurement in measurementPack:
                    measurementX.append(measurement[0])
                    measurementY.append(measurement[1])
  
            # print("before draw measurements")
            #draw measurements  
            ax.scatter(measurementX, measurementY, c = "g")

            #draw perimeters and centers
            ax.scatter(perimeterX, perimeterY, c = "m")

            # print("before draw centerX centerY")
            for centerX, centerY in zip(centerXs, centerYs):
                ax.scatter(centerX[-1], centerY[-1], c = "k")
                ax.plot(centerX, centerY, c = '#07516e', linewidth = 5.0)     
            
            # print("before auto scaling")

            #finally auto scaling the screen
            if(time == -1):
                ax.relim()
                ax.autoscale_view(True,True,True)
            
            if(mode == 1):
                time += 1
                
            mode = (mode + 1) % 2
                        
            animation.mode = mode
            animation.time = time

            animation.perimeterX = perimeterX
            animation.perimeterY = perimeterY 
            
            animation.centerXs = centerXs 
            animation.centerYs = centerYs 
            
            animation.measurementX = measurementX 
            animation.measurementY = measurementY 
            
            #end animation function
        
        
        ani = matplotlib.animation.FuncAnimation(fig, animation, 
                frames=1000, interval=2000, repeat=False) 
        
        plt.show()
        
        return ani
    else:
        print("todo")
                
        for tracker in trackers:
            
            predictions = tracker.updatedStateHistory
            
            xs = []
            Ps = []
                    
            for prediction in predictions:
                xs.append(prediction[0])
                Ps.append(prediction[1])
                
            xs = np.array(xs)
            Ps = np.array(Ps)
        
            plt.plot(xs[:,0], xs[:,1])
            showPerimeter(xs[-1][0:2], np.linalg.inv(Ps[-1][0:2,0:2]), np.pi/100, gateThreshold)        
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
