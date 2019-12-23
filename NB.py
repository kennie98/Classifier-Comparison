import math
import numpy as np


class DataSet:
    infoCategorical = 0
    infoNumerical = 1
    # information of each column, Categorical or Numerical?
    attributeInfo = [infoCategorical,  # column 1
                     infoCategorical,  # column 2
                     infoNumerical,  # column 3
                     infoNumerical,  # column 4
                     infoNumerical,  # column 5
                     infoNumerical,  # column 6
                     infoNumerical,  # column 7
                     infoNumerical,  # column 8
                     infoNumerical,  # column 9
                     infoNumerical,  # column 10
                     infoNumerical,  # column 11
                     infoNumerical,  # column 12
                     infoNumerical,  # column 13
                     infoNumerical,  # column 14
                     infoNumerical,  # column 15
                     infoNumerical,  # column 16
                     infoNumerical,  # column 17
                     infoNumerical,  # column 18
                     infoCategorical]  # column 19
    symbolPositive = 1
    symbolNegative = 0


class DataSet1:
    infoCategorical = 0
    infoNumerical = 1
    # information of each column, Categorical or Numerical?
    attributeInfo = [infoCategorical,  # column 0
                     infoNumerical,  # column 1
                     infoNumerical,  # column 2
                     infoCategorical,  # column 3
                     infoCategorical,  # column 4
                     infoCategorical]  # column 5
    symbolPositive = "private"
    symbolNegative = "public"


class NB(object):
    # define constants
    symbolMean = 'mean'
    symbolVariance = 'variance'
    symbolResultPositive = DataSet.symbolPositive
    symbolResultNegative = DataSet.symbolNegative
    infoCategorical = 0
    infoNumerical = 1
    probability_info = {}
    count_positive = count_negative = probability_positive = probability_negative = 1

    def __init__(self):
        pass

    def initial_data_calculation(self, label):
        # get the counts and probability of positive/negative
        self.count_positive = (label == self.symbolResultPositive).sum()
        self.count_negative = (label == self.symbolResultNegative).sum()
        self.probability_positive = self.count_positive / (self.count_positive + self.count_negative)
        self.probability_negative = self.count_negative / (self.count_positive + self.count_negative)

    @staticmethod
    # calculate the probability based on Normal distruction probability density function
    def normpdf(x, mean, var):
        denom = (2 * math.pi * var) ** .5
        num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
        return num / denom

    # calculate the probability of a corresponding categorical tag in the training data
    def calc_categorical_probability(self, data, label, col):
        # get all the states
        states = set(data[:, col].flatten())
        # create a dictionary to store the probability of the state
        probability = {}
        for i in states:
            probability[i] = {self.symbolResultPositive: 0, self.symbolResultNegative: 0}
            probability[i][self.symbolResultPositive] = \
                data[(label == self.symbolResultPositive) & (data[:, col] == i)].shape[0]
            probability[i][self.symbolResultNegative] = \
                data[(label == self.symbolResultNegative) & (data[:, col] == i)].shape[0]
            probability[i][self.symbolResultPositive] /= self.count_positive
            probability[i][self.symbolResultNegative] /= self.count_negative
        return probability

    # calculate the mean/variance pair of a corresponding numerical tag in the training data
    def calc_numerical_parameters(self, data, label, col):
        parameter = {}
        parameter[self.symbolResultPositive] = {self.symbolMean: 0, self.symbolVariance: 0}
        parameter[self.symbolResultPositive][self.symbolMean] = \
            data[label == self.symbolResultPositive][:, col].astype(np.float32).mean()
        parameter[self.symbolResultPositive][self.symbolVariance] = \
            data[label == self.symbolResultPositive][:, col].astype(np.float32).var(ddof=1)

        parameter[self.symbolResultNegative] = {self.symbolMean: 0, self.symbolVariance: 0}
        parameter[self.symbolResultNegative][self.symbolMean] = \
            data[label == self.symbolResultNegative][:, col].astype(np.float32).mean()
        parameter[self.symbolResultNegative][self.symbolVariance] = \
            data[label == self.symbolResultNegative][:, col].astype(np.float32).var(ddof=1)
        return parameter

    # process training data
    def training(self, data, label, parameter):
        self.initial_data_calculation(label)
        for col in range(len(DataSet.attributeInfo)):
            if DataSet.attributeInfo[col] == self.infoCategorical:
                self.probability_info[col] = self.calc_categorical_probability(data, label, col)
            else:
                self.probability_info[col] = self.calc_numerical_parameters(data, label, col)

    # prediction testing data
    def predictionResults(self, testing_x, parameter):
        probability = []
        result = []
        for i in range(testing_x.shape[0]):
            j = testing_x[i, :]
            probability.append({self.symbolResultPositive: 1, self.symbolResultNegative: 1})
            for col in range(len(DataSet.attributeInfo)):
                if DataSet.attributeInfo[col] == self.infoCategorical:
                    probability[i][self.symbolResultPositive] *= 0 if j[col] not in self.probability_info[col] else \
                        self.probability_info[col][j[col]][self.symbolResultPositive]
                    probability[i][self.symbolResultNegative] *= 0 if j[col] not in self.probability_info[col] else \
                        self.probability_info[col][j[col]][self.symbolResultNegative]
                else:
                    probability[i][self.symbolResultPositive] *= self.normpdf(j[col], \
                                                                              self.probability_info[col][
                                                                                  self.symbolResultPositive][
                                                                                  self.symbolMean], \
                                                                              self.probability_info[col][
                                                                                  self.symbolResultPositive][
                                                                                  self.symbolVariance])
                    probability[i][self.symbolResultNegative] *= self.normpdf(j[col], \
                                                                              self.probability_info[col][
                                                                                  self.symbolResultNegative][
                                                                                  self.symbolMean], \
                                                                              self.probability_info[col][
                                                                                  self.symbolResultNegative][
                                                                                  self.symbolVariance]*parameter)
            result.append(
                self.symbolResultPositive if probability[i][self.symbolResultPositive] >= probability[i][
                    self.symbolResultNegative] else self.symbolResultNegative)
        return result

    def predictionErrorRate(self, testing_x, testing_y, parameter):
        prediction = self.predictionResults(testing_x, parameter)
        error = np.sum(prediction != testing_y)
        error_rate = error / testing_y.size
        return error_rate


if __name__ == '__main__':
    import csv
    from sklearn.preprocessing import MinMaxScaler

    train_data = np.genfromtxt("train.csv", delimiter=',', dtype="|U15" )
    test_data = np.genfromtxt("test.csv", delimiter=',', dtype="|U15")
    train_data = np.delete(train_data, (0), axis=0)
    test_data = np.delete(test_data, (0), axis=0)
    data = np.concatenate((train_data, test_data))
    data_dim = data.shape[1]

    # normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    col2 = data[:, 1:3]
    col2 = scaler.fit_transform(col2.astype(np.float32))
    data[:, 1:3] = col2
    data[:, 1:3].astype(np.float32)

    train = data[:20, :]
    test = data[20:, :]
    train_x = train[:, :data_dim - 1]
    train_y = train[:, data_dim - 1]
    test_x = test[:, :data_dim - 1]
    test_y = test[:, data_dim - 1]

    nb = NB()
    nb.training(train_x, train_y, None)
    print('NB Error Rate:' + str(nb.predictionErrorRate(test_x, test_y, None)))

    print("done")
