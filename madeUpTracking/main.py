    # -*- coding: utf-8 -*-
    
    import objectCreator as oC
    from track_singleTarget_singleModel import Tracker_SingleTarget_SingleModel
    from track_singleTarget_multipleModel import Tracker_SingleTarget_MultipleModel
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    #%matplotlib qt
    
    plot = True
    demo_mode = 1
    mixDemos = [0,1]
    runPrevious = False
    std = 2
    
    """
        mode 0: track_singleTarget_singleModel
        mode 1: track_singleTarget_multipleModel
        mode 2: track_multipleTarget_singleModel
        mode 3: track_multipleTarget_multipleModel
    """
    
    
    def calculateL2Error(originalObject, predictionObject): 
        
        return np.sum(np.sqrt(  np.square(originalObject[:,0] - predictionObject[:,0]) + \
                                np.square(originalObject[:,1] - predictionObject[:,1])))
        
    
    if(not runPrevious):
        (originalObjectPaths, noisyObjectPaths) = oC.generateObjectPaths(objectCount = 2, std = std, pointDistance = 0.5, frame = {"w":100,"h":120}, pathInnerPointCount=3)
        noisyObject0 = noisyObjectPaths[0]
        originalObject0 = originalObjectPaths[0]
    
    
    
    # plt.plot(noisyObject0[:,0], noisyObject0[:,1])
    # plt.plot(originalObject0[:,0], originalObject0[:,1])
    
    if(0 in mixDemos):
        
        tracksToShow = [True, False, True, False] # classic kalman, constantTurnRate, constantVelocity, RandomMotion
    
    
        tracker1 = Tracker_SingleTarget_SingleModel(modelType = 0, deltaT = 0.1, measurementNoiseStd = std)
        tracker2 = Tracker_SingleTarget_SingleModel(modelType = 1, deltaT = 0.1, measurementNoiseStd = std)
        tracker3 = Tracker_SingleTarget_SingleModel(modelType = 2, deltaT = 0.1, measurementNoiseStd = std)
        tracker4 = Tracker_SingleTarget_SingleModel(modelType = 3, deltaT = 0.1, measurementNoiseStd = std)
    
        for i,point in enumerate(noisyObject0):
            #print(i)
            tracker1.predictAndUpdate(point)
            tracker2.predictAndUpdate(point)
            tracker3.predictAndUpdate(point)
            tracker4.predictAndUpdate(point)
            
    
        predictions1 = np.array(tracker1.updatedPredictions).reshape(len(tracker1.updatedPredictions), 2)
        predictions2 = np.array(tracker2.updatedPredictions).reshape(len(tracker2.updatedPredictions), 2)
        predictions3 = np.array(tracker3.updatedPredictions).reshape(len(tracker3.updatedPredictions), 2)
        predictions4 = np.array(tracker4.updatedPredictions).reshape(len(tracker4.updatedPredictions), 2)
        
    
        if(plot):
            
            plt.figure()
            
            if(tracksToShow[0]):
                
                plt.plot(predictions1[:,0], predictions1[:,1], 'r', linewidth=2)
                plt.plot(predictions1[-100:,0], predictions1[-100:,1], 'r', linewidth=3.5)
                plt.scatter(predictions1[100,0], predictions1[100,1], linewidths=10, c='r' )
                
            
            if(tracksToShow[1]):
                
                plt.plot(predictions2[:,0], predictions2[:,1], 'k', linewidth=1.5)
                plt.plot(predictions2[-100:,0], predictions2[-100:,1], 'k', linewidth=3.5)
                plt.scatter(predictions2[100,0], predictions2[100,1], linewidths=10, c='k' )
                
    
            if(tracksToShow[2]):
            
                plt.plot(predictions3[:,0], predictions3[:,1], 'c', linewidth=1.5)
                plt.plot(predictions3[-100:,0], predictions3[-100:,1], 'c', linewidth=3.5) 
                plt.scatter(predictions3[100,0], predictions3[100,1], linewidths=10, c='c' )
                
            if(tracksToShow[3]):
            
                plt.plot(predictions4[:,0], predictions4[:,1], color = "#a5a03b", linewidth=1.5)
                plt.plot(predictions4[-100:,0], predictions4[-100:,1], color = "#a5a03b", linewidth=3.5) 
                plt.scatter(predictions4[100,0], predictions4[100,1], linewidths=10, color = "#a5a03b" )                
                
                
            plt.plot(originalObject0[:,0], originalObject0[:,1], 'b', linewidth=2)
            plt.plot(originalObject0[-100:,0], originalObject0[-100:,1], 'b', linewidth=3.5)
            
            plt.scatter(noisyObject0[:,0], noisyObject0[:,1], c="g", linewidth=0.1)
            
            plt.scatter(originalObject0[100,0], originalObject0[100,1], linewidths=10, c='b' )

            
    
        print(calculateL2Error(originalObject0[100:,:], predictions1[100:,:]))
        print(calculateL2Error(originalObject0[100:,:], predictions2[100:,:]))    
        print(calculateL2Error(originalObject0[100:,:], predictions3[100:,:]))
        print(calculateL2Error(originalObject0[100:,:], predictions4[100:,:]))
        
    
    
        # plt.plot((predictions - originalObject0)[:,0])
        # plt.plot((predictions - originalObject0)[:,1])
    
        # plt.plot(predictions[:,0], predictions[:,1], linestyle='none', marker='o')
    
    
    if(1 in mixDemos):
        tracker1 = Tracker_SingleTarget_MultipleModel(deltaT = 0.1, measurementNoiseStd = std)
        
        for point in noisyObject0:
            tracker1.predictAndUpdate(point)
    
    
        predictions1 = np.array(tracker1.updatedPredictions).reshape(len(tracker1.updatedPredictions), 2)
        mus = np.array(tracker1.mus)
    
    
        if(plot):
            
            plt.figure()
    
            plt.plot(predictions1[:,0], predictions1[:,1], 'm', linewidth=2)
            plt.plot(predictions1[-100:,0], predictions1[-100:,1], 'm', linewidth=3.5)
            
    
            plt.plot(originalObject0[:,0], originalObject0[:,1], 'b', linewidth=2)
            plt.plot(originalObject0[-100:,0], originalObject0[-100:,1], 'b', linewidth=3.5)
            
            plt.scatter(noisyObject0[:,0], noisyObject0[:,1], c="g", linewidth=0.1)
            
            plt.scatter(originalObject0[100,0], originalObject0[100,1], linewidths=10, c='g' )
            
            plt.figure()
            plt.plot(mus[:,0], c = "c")
            plt.plot(mus[:,1], c = "k")
            if(mus.shape[1] > 2):
                plt.plot(mus[:,2], c = "r")
            
            
        print(mus.shape)
    
        print(calculateL2Error(originalObject0[100:,:], predictions1[100:,:]))
        print("-")
    elif(demo_mode == 2):
        print("Not implemented yet")
    elif(demo_mode == 3):
        print("Not implemented yet")
    
    
    
    
    
    
    
