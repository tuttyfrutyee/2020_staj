# OPTIMIZATION
I had little time to spend on optimization of the process noise. I have explained why it is necessary to optimize process noise in my report. Here this folder contains my attempt to optimization. I have used the technique presented here [Probabilistic 3D Multi-Object Tracking for Autonomous Driving, 2020]. They basically estimate the process noise by using the models and ground-truth data. On top of this, I have discovered that scaling this obtained process noise helps to get better results. 

## Initial Point Finder

It has the code to estimate process noise from ground-truth data using the technique ginve in [Probabilistic 3D Multi-Object Tracking for Autonomous Driving, 2020].

## Optimizers

My first attempt was to learn the parameters of process noise by gradient-descent, which gave me bad results because it almost always give local minimum. Also it takes to much time to back-propagate from UKF. Hence I do not use it anymore. But anyhow, I have implemented the same tracker classes in "madeUpTracking" here with torch, I have not changed it in case someone needs it.

My last call is to first find initial process noise by "InitialPointFinder" and then visualize the loss space by multiplying this initial process noise with scales picked from an intervel to choose the best scale. Then use initialPoint_processNoise * best_scale as your process noise.

I had tried also back-propogate this scale variable, but it is realy not necessary and again takes too much time for UKF. Just pick same scale points and then visualize the loss space and decide from there for the best scale.
