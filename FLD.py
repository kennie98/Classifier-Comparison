import numpy as np
import Helper


class FLD(object):
    helper = None
    w = None

    def __init__(self):
        self.helper = Helper.Helper()
        pass

    def training(self, training_x, training_y, parameter):
        # assume that we are always dealing with binary classification problem
        x1, x2 = self.helper.separate(training_x, training_y)

        # calculate mean vectors
        u1 = x1.mean(axis=0)
        u2 = x2.mean(axis=0)

        # remove means from classes
        x1mc = x1 - u1
        x2mc = x2 - u2

        # calculate covariance matrix
        S1 = np.dot(x1mc.T, x1mc)
        S2 = np.dot(x2mc.T, x2mc)
        Sw = S1 + S2

        self.w = np.dot(np.linalg.inv(Sw), (u1 - u2))

    def predictionResults(self, testing_x, threshold):
        return (1 - np.sign(np.dot(self.w, testing_x.T) + threshold)) / 2

    def predictionErrorRate(self, testing_x, testing_y, threshold):
        prediction = self.predictionResults(testing_x, threshold)
        error = np.sum(prediction != testing_y)
        error_rate = error / testing_y.size
        return error_rate