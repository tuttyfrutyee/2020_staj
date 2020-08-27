import torch
import Tracker
import torch.optim as optim

import sys
sys.path.append("../../Data/Scenarios")
sys.path.append("../../Data/Train")

import trainData 

dataPacks = trainData.dataPacks


dtype_torch = torch.float64
torch.autograd.set_detect_anomaly(True)

#helper functions

def calculateLoss(groundTruth, predictionMeasured):
    
    diff = groundTruth - predictionMeasured
    
    return torch.mm(diff.T, diff)



#to optimize

#########################################

scale = torch.tensor([
    1
], dtype= dtype_torch, requires_grad=True)


def getProcessNoiseCov():

    return    torch.tensor([
       [0.12958419304911287,0,0,0,0],
       [0,0.20416385918814656,0,0,0],
       [0,0,0.008794949000079913,0,0],
       [0,0,0,0.8057826337426066,0],
       [0,0,0,0,0] 
     
    ], dtype=dtype_torch) / scale



#########################################

    

learningRate = 1e-3



optimizer = optim.Adam([scale], lr = learningRate)

Tracker.ProcessNoiseCov = getProcessNoiseCov()


sequenceLength = 10


dt = 0.1

losses_ = []

for k in range(10):

    losses = []

    for s, dataPack in enumerate(dataPacks):
        
        lossTotal = 0
        loss = 0
        
        tracker = Tracker.Tracker_SingleTarget_SingleModel_CV_allMe(0)
        
        measurementPacks, groundTruthPacks = dataPack
        
        for i, (measurementPack, groundTruthPack) in enumerate(zip(measurementPacks, groundTruthPacks)):
            
            if(i != 0 and (i % sequenceLength == 0 or i == len(measurementPacks) - 1)):

                z = tracker.feedMeasurement(torch.from_numpy(measurementPack[0]), dt)
                if(z is not None):
                    loss += calculateLoss(torch.from_numpy(groundTruthPack[0]), z)
                
                loss.backward()
                lossTotal += loss.item()
                loss = 0
                
                optimizer.step()
                optimizer.zero_grad()

                tracker.detachTrack()

                Tracker.ProcessNoiseCov = getProcessNoiseCov()
                                                   
                
            else:
                z = tracker.feedMeasurement(torch.from_numpy(measurementPack[0]), dt)
                if(z is not None):
                    loss += calculateLoss(torch.from_numpy(groundTruthPack[0]), z)

        if(s == 0):
            print(lossTotal)
            
        losses.append(lossTotal)

    losses_.append(losses)
                

               
            
    







