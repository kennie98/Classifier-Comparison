import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics


class Helper:
    # separate x according to values in y
    def separate(self, x, y):
        s = list(set(y.flatten()))
        x1 = x[y == s[0]]
        x2 = x[y == s[1]]
        return x1, x2

    def frange(self, x, y, jump):
        while x < y:
            yield x
        x += jump

    def crossValidation(self, data_x, data_y, batches, parameter, classifier):
        avg_err_rate = 0
        count = 0
        for i in range(batches):
            testing_x = np.array(data_x[i])[0]
            testing_y = np.array(data_y[i])[0]

            training_x = np.concatenate([data_x[j] for j in range(batches) if j != i], axis=0)
            training_y = np.concatenate([data_y[j] for j in range(batches) if j != i], axis=0).flatten()
            training_x = np.reshape(training_x, (training_x.shape[0] * training_x.shape[1], training_x.shape[2]))

            classifier.training(training_x, training_y, parameter)
            err_rate = classifier.predictionErrorRate(testing_x, testing_y, parameter)
            avg_err_rate += err_rate
            count += 1
        avg_err_rate /= count
        return avg_err_rate

    def getOptimalParameter(self, error_list):
        npel = np.array(error_list).T[1]
        index = np.where(npel == np.amin(npel))
        i = int((index[0].shape[0] - 1) / 2)
        i = index[0][i]
        return error_list[i][0]

    def plotErrorRateCurve(self, error_list, title, xlabel):
        xdata = np.array(error_list).T[0].tolist()
        ydata = np.array(error_list).T[1].tolist()
        plt.figure()
        plt.plot(xdata, ydata)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Error Rate')
        plt.ion()
        plt.show()

    def plotConfusionMatrix(self, xlabels, prediction, title):
        # print out the confusion matrix
        labels = ['no sign of DR', 'signs of DR']
        cm = confusion_matrix(xlabels, prediction)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, z, ha='center', va='center')
        plt.title(title)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.ion()
        plt.show()

    def plotRocCurve(self, xlabels, prediction, title):
        fpr, tpr, threshold = metrics.roc_curve(xlabels, prediction)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title(title)
        plt.ion()
        plt.show()

#        fprl = []
#        tprl = []
#        for p in prediction:
#            fpr, tpr, threshold = metrics.roc_curve(xlabels, p)
#            fprl.append(fpr[1])
#            tprl.append(tpr[1])
#        plt.figure()
#        plt.title("ROC Curves")
#        plt.plot(fprl, tprl, label=title)
#        plt.plot([0, 1], [0, 1], 'r-')
#        plt.legend(loc=4)
#        plt.xlim([0, 1])
#        plt.ylim([0, 1])
#        plt.ion()
#        plt.show()
