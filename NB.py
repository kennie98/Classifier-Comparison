import math
import numpy as np

class data:
    infoCategorical = 0
    infoNumerical = 1
    # information of each column, Categorical or Numerical?
    attributeInfo = [infoCategorical,   # column 1
                     infoCategorical,   # column 2
                     infoNumerical,     # column 3
                     infoNumerical,     # column 4
                     infoNumerical,     # column 5
                     infoNumerical,     # column 6
                     infoNumerical,     # column 7
                     infoNumerical,     # column 8
                     infoNumerical,     # column 9
                     infoNumerical,     # column 10
                     infoNumerical,     # column 11
                     infoNumerical,     # column 12
                     infoNumerical,     # column 13
                     infoNumerical,     # column 14
                     infoNumerical,     # column 15
                     infoNumerical,     # column 16
                     infoNumerical,     # column 17
                     infoNumerical, ]   # column 18

class NB(object):
    # define constants
    symbolMean = 'mean'
    symbolVariance = 'variance'
    symbolPublic = 'public'
    symbolPrivate = 'private'
    infoCategorical = 0
    infoNumerical = 1
    probability_info = {}
    count_private = count_public = probability_private = probability_public = 1


    def __init__(self):
        # get the counts and probability of private/public
        self.count_private = train_data[self.tagPrivatePublic].value_counts()[self.symbolPrivate]
        self.count_public = train_data[self.tagPrivatePublic].value_counts()[self.symbolPublic]
        self.probability_private = self.count_private / (self.count_private + self.count_public)
        self.probability_public = self.count_public / (self.count_private + self.count_public)

    @staticmethod
    # calculate the probability based on Normal distruction probability density function
    def normpdf(x, mean, var):
        denom = (2 * math.pi * var) ** .5
        num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
        return num / denom

    # calculate the probability of a corresponding categorical tag in the training data
    def calc_categorical_probability(self, tag):
        # get all the states
        states = set(train_data[tag].tolist())
        # create a dictionary to store the probability of the state
        probability = {}
        for i in states:
            probability[i] = {self.symbolPrivate: 0, self.symbolPublic: 0}
            for j in range(len(train_data[tag])):
                if train_data[tag][j] == i:
                    if train_data[self.tagPrivatePublic][j] == self.symbolPrivate:
                        probability[i][self.symbolPrivate] += 1
                    else:
                        probability[i][self.symbolPublic] += 1
            probability[i][self.symbolPrivate] /= self.count_private
            probability[i][self.symbolPublic] /= self.count_public
        return probability

    # calculate the mean/variance pair of a corresponding numerical tag in the training data
    def calc_numerical_parameters(self, tag):
        parameter = {}
        parameter[self.symbolPrivate] = {self.symbolMean: 0, self.symbolVariance: 0}
        parameter[self.symbolPrivate][self.symbolMean] = \
        train_data.loc[train_data[self.tagPrivatePublic] == self.symbolPrivate][tag].mean()
        parameter[self.symbolPrivate][self.symbolVariance] = \
        train_data.loc[train_data[self.tagPrivatePublic] == self.symbolPrivate][tag].var()
        parameter[self.symbolPublic] = {self.symbolMean: 0, self.symbolVariance: 0}
        parameter[self.symbolPublic][self.symbolMean] = \
        train_data.loc[train_data[self.tagPrivatePublic] == self.symbolPublic][tag].mean()
        parameter[self.symbolPublic][self.symbolVariance] = \
        train_data.loc[train_data[self.tagPrivatePublic] == self.symbolPublic][tag].var()
        return parameter

    # process training data
    def process_training_data(self):
        for key, value in self.attributeInfo.items():
            if value == self.infoCategorical:
                self.probability_info[key] = self.calc_categorical_probability(key)
            else:
                self.probability_info[key] = self.calc_numerical_parameters(key)

    # prediction testing data
    def process_testing_data(self):
        probability = []
        result = []
        print("== Bayesian Probabilities: ==")
        for i, r in test_data.iterrows():
            probability.append({self.symbolPrivate: 1, self.symbolPublic: 1})
            for key, value in self.attributeInfo.items():
                if value == self.infoCategorical:
                    probability[i][self.symbolPrivate] *= 0 if r[key] not in self.probability_info[key] else \
                    self.probability_info[key][r[key]][self.symbolPrivate]
                    probability[i][self.symbolPublic] *= 0 if r[key] not in self.probability_info[key] else \
                    self.probability_info[key][r[key]][self.symbolPublic]
                else:
                    probability[i][self.symbolPrivate] *= self.normpdf(r[key],
                                                                       self.probability_info[key][self.symbolPrivate][
                                                                           self.symbolMean],
                                                                       self.probability_info[key][self.symbolPrivate][
                                                                           self.symbolVariance])
                    probability[i][self.symbolPublic] *= self.normpdf(r[key],
                                                                      self.probability_info[key][self.symbolPublic][
                                                                          self.symbolMean],
                                                                      self.probability_info[key][self.symbolPublic][
                                                                          self.symbolVariance])
            result.append(self.symbolPrivate if probability[i][self.symbolPrivate] >= probability[i][
                self.symbolPublic] else self.symbolPublic)
            print(probability[i])
        print("\n== Prediction Result: ==")
        print(result)

        # print out the confusion matrix
        labels = [self.symbolPrivate, self.symbolPublic]
        cm = confusion_matrix(test_data[self.tagPrivatePublic], pd.DataFrame(result), labels)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, z, ha='center', va='center')
        plt.title('Confusion Matrix of The Classifier')
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()


if __name__ == '__main__':
    assignment2 = assignment2()
    assignment2.process_training_data()
    assignment2.process_testing_data()
    print("done")
