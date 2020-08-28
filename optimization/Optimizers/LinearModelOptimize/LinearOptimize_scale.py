import torch
import Tracker
import torch.optim as optim
import math

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
        [ 0.0251,  0.0219, -0.0072, -0.0054],
        [ 0.0219,  0.0199, -0.0064, -0.0049],
        [-0.0072, -0.0064,  0.0023,  0.0016],
        [-0.0054, -0.0049,  0.0016,  0.0017]
     
    ], dtype=dtype_torch) * scale



#########################################

    

learningRate = 1

sequenceLength = 50


optimizer = optim.Adam([scale], lr = learningRate / sequenceLength)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 10, factor=0.1, verbose=True)

Tracker.ProcessNoiseCov = getProcessNoiseCov()




dt = 0.1

minimumSliceCount = float("inf")
batchSize = 10
losses_ = []

for dataPack in dataPacks:
    sliceCount = int(len(dataPack[0]) / sequenceLength)
    if(minimumSliceCount > sliceCount):
        minimumSliceCount = sliceCount

for epoch in range(100):
    
    losses = []
    trackers = []
    
    for _ in dataPacks:
        trackers.append(Tracker.Tracker_SingleTarget_SingleModel_Linear_allMe())
    
    for s in range(minimumSliceCount):
        
        totalLoss = 0
        
        batches = []
        
        for b in range(math.ceil(len(dataPacks)/batchSize)):
            
            if((b+1)*batchSize > batchSize):
                batches.append(dataPacks[b*batchSize : ])
            else:
                batches.append(dataPacks[b*batchSize : (b+1)*batchSize])
                
        for b, batch in enumerate(batches):       
            
            loss = 0
        
            for l, dataPack in enumerate(batch):
                
                tracker = trackers[b*batchSize + l]
                
                measurementPacks = dataPack[0][s*sequenceLength: (s+1) * sequenceLength]
                groundTruthPacks = dataPack[1][s*sequenceLength: (s+1) * sequenceLength]
                
                for i, (measurementPack, groundTruthPack) in enumerate(zip(measurementPacks, groundTruthPacks)):
                    
                    z = tracker.feedMeasurement(torch.from_numpy(measurementPack[0]), dt)
                    if(z is not None):
                        loss += calculateLoss(torch.from_numpy(groundTruthPack[0]), z) / len(dataPacks)
                
            loss.backward()
            totalLoss += loss.item()
            
            optimizer.step()
            optimizer.zero_grad()
            
            for l, _ in enumerate(batch):                
                tracker = trackers[b*batchSize + l]                
                tracker.detachTrack()
                
            Tracker.ProcessNoiseCov = getProcessNoiseCov()
        
        print(totalLoss)
        print(scale)
        scheduler.step(totalLoss)
        losses.append(totalLoss)

    
    losses_.append(losses)
    
    

# losses_ = []

# for k in range(100):

#     losses = []

#     for s, dataPack in enumerate(dataPacks):
        
#         lossTotal = 0
#         loss = 0
        
#         tracker = Tracker.Tracker_SingleTarget_SingleModel_Linear_allMe()
        
#         measurementPacks, groundTruthPacks = dataPack
        
#         for i, (measurementPack, groundTruthPack) in enumerate(zip(measurementPacks, groundTruthPacks)):
            
#             if(i != 0 and (i % sequenceLength == 0 or i == len(measurementPacks) - 1)):

#                 z = tracker.feedMeasurement(torch.from_numpy(measurementPack[0]), dt)
#                 if(z is not None):
#                     loss += calculateLoss(torch.from_numpy(groundTruthPack[0]), z)
                
#                 loss.backward()
#                 lossTotal += loss.item()
#                 loss = 0
                
#                 optimizer.step()
#                 optimizer.zero_grad()

#                 tracker.detachTrack()

#                 Tracker.ProcessNoiseCov = getProcessNoiseCov()
                                                   
                
#             else:
#                 z = tracker.feedMeasurement(torch.from_numpy(measurementPack[0]), dt)
#                 if(z is not None):
#                     loss += calculateLoss(torch.from_numpy(groundTruthPack[0]), z)

#         if(s == 0):
#             scheduler.step(lossTotal)
#             print(lossTotal)
#             print("s = 0", scale)
#         elif(s==1):
#             print("s = 1", scale)
#         elif(s==2):
#             print("s = 2", scale)            
            
#         losses.append(lossTotal)

#     losses_.append(losses)
                

               
            
    







