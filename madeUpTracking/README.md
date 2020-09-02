# TRACKER DEMOS

 There are 3 kind of demos here. Each of them use scenarios from Scenarios for generation of mock data for demo purposes.

I will now explain what each of those demos do.

## DEMO SINGLE TARGET

First question one might ask why do you need a "single target demo"? It was for developing purposes, and in fact it is very usefull when trying out different process noises for models individually to test on one target.

### HOW IT WORKS?

First one should make sure that the working directory of the python interpreter is same as the demo file. Demo obviously import some other code I have written which has an order.

Then just run the file, and hopefully some graph will show some scenario and the trackers result of tracking it.

You can change the scenario to run the tracker just by selecting different scenario from the scn object. Go checkout the file scenario.py inside the Scenarios to see what scenarios exists and how to create new scenarios.


After making sure what scenario to work on, you have 5 options at total for tracker model. Here is the list:

#### IF imm = True:
CV & CVTR fused models will be used no matter what modeType is selected

#### IF imm = False: 
Each different models can be selected by setting "modeType"

#### ***Mode 0***:
Linear model, **Constant velocity, No Turn Rate** (Classical Kalman Filter)

#### ***Mode 1***:
Non-Linear model, **Constant velocity, No Turn Rate** (CV) (For Unscented Kalman Filter)

#### ***Mode 2***:
Non-Linear model, **Constant velocity, Constant Turn Rate** (CVTR)  (For Unscented Kalman Filter)

#### ***Mode 3***:
Non-Linear model, **Random Motion** (RM) (For Unscented Kalman Filter)


### PLOTTING

If imm is selected then mode probabilities will be shown additional to the scenerio plot and tracker estimates.



## DEMO MULTIPLE TARGET - SINGLE MODEL



### HOW IT WORKS?

First one should make sure that the working directory of the python interpreter is same as the demo file. Demo obviously import some other code I have written which has an order.

Then just run the file, and hopefully some graph will show some scenario and the trackers result of tracking it. If real time plotting is open the results will be shown with some interval. This feature I have added was for debugging purposes and to understand the trackers better what they do in each step.

You can change the scenario to run the tracker just by selecting different scenario from the scn object. Go checkout the file scenario.py inside the Scenarios to see what scenarios exists and how to create new scenarios.


After making sure what scenario to work on, you have 4 options at total for tracker model. Here is the list:


#### IF imm = False: 
Each different models can be selected by setting "modeType"

#### ***Mode 0***:
Linear model, **Constant velocity, No Turn Rate** (Classical Kalman Filter)

#### ***Mode 1***:
Non-Linear model, **Constant velocity, No Turn Rate** (CV) (For Unscented Kalman Filter)

#### ***Mode 2***:
Non-Linear model, **Constant velocity, Constant Turn Rate** (CVTR)  (For Unscented Kalman Filter)

#### ***Mode 3***:
Non-Linear model, **Random Motion** (RM) (For Unscented Kalman Filter)


### PLOTTING

For multiple target tracking I have made some modifications in the tracker objects code for storing the history of the tracker. Using this history we can plot the results as if they are playing in real time. To achieve this inside the myHelpers folder there is visualizationHelper.py file. You can change the speed of the visualizaiton from there. If you do not want to show results in real time just change second last parameter from the visualizeTrackingResults to False.


## DEMO MULTIPLE TARGET - MULTIPLE MODEL

To be honest my first impression is single model's results are better than imm, but i was able to get better results with imm with single target. Probably tweaking the parameters(spatialDensity, PD, etc..) and optimizing process noises would give eventually better results with multiple model on multiple targets. In fact it is not all that bad. It allows non oscillating tracks on linear lines and at the same time can return corners without being too overconfident about going straight. It is indeed promising but i did not have enough time to play with it.



But no matter what, here is the demo. Everythin is same as the MULTIPLE TARGET SINGLE MODEL demo.
