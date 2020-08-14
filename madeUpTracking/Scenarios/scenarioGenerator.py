import objectGenerator.objectGenerator as oG
import matplotlib.pyplot as plt

class Scenario(object):
    
    def __init__(self, stds, objectPathCorners, corruptions, stepSizes, colors):
        
        self.objects = []
        
        assert(len(stds) == len(objectPathCorners) == len(stepSizes) == len(colors))
        
        for i,objectPointCorners in enumerate(objectPathCorners):
            
            std = stds[i]
            color = colors[i]
            stepSize = stepSizes[i]
            
            if(corruptions[i]):
                corruptionCount, corruptionLength = corruptions[i]
            else:
                corruptionCount = None
                corruptionLength = None
                
            if(not color):
                color = ["b", "g"]
            if(not stepSize):
                stepSize = 0.5
            if(not std):
                std = 4
            
            if(objectPointCorners == None):
                
                #generate object points randomly      
    
                newObject = oG.Object_(color, None, None, 4, stepSize)

                newObject.generateNoisyMeasurements(std, corruptionCount, corruptionLength)                
                
            else:
                
                newObject = oG.Object_(color, objectPointCorners[0], objectPointCorners[1], None, stepSize)

                newObject.generateNoisyMeasurements(std, corruptionCount, corruptionLength)
                
                
            
            self.objects.append(newObject)
            
        
    def plotScenario(self):
        
        plt.figure()
        
        for object_ in self.objects:
            object_.drawObject(1)
        