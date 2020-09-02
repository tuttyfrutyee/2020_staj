# SCENARIOS

This folder's main purpose is to create mock data to test trackers.

## HOW DOES IT WORK?

Inside the scenarios.py file you write your own scenario, and the choose it in the tracker demo you want to test it. Note that each scenario has a function to draw the contents of it's own. So do not forget to plot right scenario at the bottom of the tracker demo script. 

Note that each demo has two function at the top of their scripts to get the data needed from the scenarios.


## HOW TO CREATE A NEW SCENARIO?

You need to generate a scenario object using the scenarioGenerator object.
It has 5 different parameters to initialize with.

Here is the list:

### stds, objectPathCorners, corruptions, stepSizes, colors

#### 1) stds : List
It is the standart deviation of the gausian noise that will be added to the generated path, which are the measurements this scenario will have. It is a list. If there is one object you want to generate then just put one std into the list, like [1.2] or if you want three objects, then [1.3, None, 1]. If the value of the std is None, then the default std value inside the scenarioGenerator will be used.

#### 2) objectPathCorners: List

It is the corners of the objects' path that you want it to accross. The number of corners will change the complexity and the length of the path you are generating. The generation of the paths is thanks to the spline code that I have obtained from Gorkem. You can ask him about it if you want it.

Again it is a list, actually it will be tuples in list. Why tuples, because we need both x and y coordinates. They need to provided seperately

Here is an example:

One object with passing through (x=5, y=7), (x=23, y=18) and (x=30, y=2)

#### [&nbsp; ( [5, 23, 30], [7, 18, 2] ) &nbsp;]

Another example for three objects:

#### [&nbsp; None, &nbsp; ***(*** [11,27,34], [ 65, 86, 12 ] ***)***, &nbsp; ***(*** [ 5, 23, 30], [ 7, 18, 2 ***])*** &nbsp; ]

None means generate the path corners randomly. It is very handy for random scenario generation.

#### 3) corruptions: List
It is for disabling measurements for some part of the objects' path randomly. It is good for simulating the effect of missed measurement.

Here is an example of 3 number of corruption and each with length 4

[&nbsp; (3,4),&nbsp; ]

And another one with two objects that we don't want any corruptions in their measurements

[&nbsp; None, None ]

Note that this corruption is only for measurement, the groundTruth values won't be affected, only measurements are affected.

#### 4) stepSizes: List

stepsizes change the density of the points generated in the path.

Example : 

[0.4, 0.4, None]

Again None means selecting the default value that is in scenarioGenerator

#### 5) colors : List

I thought it would be cool to assign different colors to the objects. It turns out that it was not much usefull, just put **None** to the elements of the list and matplotlib will randomly generate colors for the objects.

Example : [&nbsp; None,&nbsp; None, None ]





