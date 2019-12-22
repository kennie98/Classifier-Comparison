'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
import numpy as np
import operator
import Helper


class KNN(object):
    helper = None
    dataSet = None
    labels = None

    def __init__(self):
        self.helper = Helper.Helper()
        pass

    def classify0(self, inX, k):
        dataSetSize = self.dataSet.shape[0]
        diffMat = np.tile(inX, (dataSetSize, 1)) - self.dataSet
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        sortedDistIndicies = distances.argsort()
        classCount = {}
        for i in range(k):
            voteIlabel = self.labels[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def training(self, training_x, training_y, k):
        self.dataSet = training_x
        self.labels = training_y

    def predictionResults(self, testing_x, k):
        results = []
        for i in testing_x:
            results.append(self.classify0(i, k))
        return results

    def predictionErrorRate(self, testing_x, testing_y, threshold):
        prediction = self.predictionResults(testing_x, threshold)
        error = np.sum(prediction != testing_y)
        error_rate = error / testing_y.size
        return error_rate
