# -*- coding: utf-8 -*-


class Tracker_MultipleTarget_SingleModel(object):


    """
        This class uses only constant velocity linear model(hence the name single model)
    """

    def __init__(self, deltaT, measurementNoiseStd ):

        self.trackers = []


    def predict(self):

        """
            Predicts the next state for all existing tracks

            Returns : None
        """

        print("todo : predict")

    def update(self, )

    def findLeftOverMeasurements(self, measurements):
        """
            It greedy matches tracks and measurements based on mahalonobis distance
            Returns the ones the measurements that could not be matched with any existing track

            Returns : leftOverMeasurements

            Note : Use it after called self.predict()
        """

        print("todo : findLeftOverMeasurements")


    def initTracksFromLonelyMeasurements(self, measurements):

        """
            Using the measurements that are left over from the greed match, we init new tracks

            Returns : None
        """

        print("todo : initTracksFromLonelyMeasurements")


    def pruneTracks(self):

        """
            Some tracks might refer to the same object in real life.
            This function gets rid of the duplicates

            Returns : None
        """


        print("todo : pruneTracks")