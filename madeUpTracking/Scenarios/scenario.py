# -*- coding: utf-8 -*-
import sys
sys.path.append("./Scenarios")
import scenarioGenerator as scenario
import numpy as np


"""

    Scenario 0: Single object with no corruptions

"""


stds_0 = [np.sqrt(4)]
objectPathCorners_0 = [([55, 23, 51], [56,9,20])] 

# objectPathCorners_0 = [(None)] 

corruptions_0 = [None]
stepSizes_0 = [0.4]
colors_0 = [("b", "g")]

scenario_0 = scenario.Scenario(stds_0, objectPathCorners_0, corruptions_0, stepSizes_0, colors_0)

#scenario_0.plotScenario()


a = 3

######################################################

"""

    Scenario 1: Single object with corruptions

"""
stds_1 = [3]
objectPathCorners_1 = [([55, 23, 51], [56,9,20])] 
corruptions_1 = [(3, 5)]
stepSizes_1 = [0.4]
colors_1 = [("b", "g")]

scenario_1 = scenario.Scenario(stds_1, objectPathCorners_1, corruptions_1, stepSizes_1, colors_1)

#scenario_1.plotScenario()


######################################################

"""

    Scenario 2: 2 object with no corruptions

"""

stds_2 = [1, 1]
objectPathCorners_2 = [ ([70, 23, 51], [0,59,20]), ([70, 56, 40], [0,-10,70]) ] 
corruptions_2 = [None, None]
stepSizes_2 = [0.4, 0.4]
colors_2 = [("b", "g"), ("k", "c")]

scenario_2 = scenario.Scenario(stds_2, objectPathCorners_2, corruptions_2, stepSizes_2, colors_2)

#scenario_2.plotScenario()






######################################################


"""

    Scenario 3: 2 object with corruptions

"""
stds_3 = [3, 2]
objectPathCorners_3 = [ ([55, 23, 51], [56,9,20]), None ] 
corruptions_3 = [None, None]
stepSizes_3 = [0.4, None]
colors_3 = [("b", "g"), None]

scenario_3 = scenario.Scenario(stds_3, objectPathCorners_3, corruptions_3, stepSizes_3, colors_3)

#scenario_3.plotScenario()




######################################################

"""

    Scenario 4: ?

"""
######################################################
