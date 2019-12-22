import numpy as np
import Helper
from sklearn.preprocessing import MinMaxScaler
import FLD
import KNN
import SVM
from datetime import datetime, timedelta

TRAINNING_DATA_SIZE = 840  # make sure that this is divisible by CV_DATA_GROUP
CROSS_VALIDATION_DATA_GROUP = 4
CROSS_VALIDATION_DATA_SIZE = (TRAINNING_DATA_SIZE / CROSS_VALIDATION_DATA_GROUP)
MAX_MICROSECOND = 99999


def frange(x, y, jump):
    while x + jump <= y:
        yield x
        x += jump


def getTimeDifference(start):
    now = datetime.now()  # .microsecond
    return now - start #if (now > start) else (now + MAX_MICROSECOND - start)


def readData(csvfile):
    # read in data
    data = np.genfromtxt(csvfile, delimiter=',')
    data_dim = data.shape[1]

    # normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data.astype('float32'))

    # separate training and dataset
    np.random.shuffle(scaled)
    train = scaled[:TRAINNING_DATA_SIZE, :]
    test = scaled[TRAINNING_DATA_SIZE:, :]
    train_x = train[:, :data_dim - 1]
    train_y = train[:, data_dim - 1]
    train_cv_x = []
    train_cv_y = []

    for i in range(CROSS_VALIDATION_DATA_GROUP):
        train_cv_x.append([train_x[int(i * CROSS_VALIDATION_DATA_SIZE):int((i + 1) * CROSS_VALIDATION_DATA_SIZE), :]])
        train_cv_y.append([train_y[int(i * CROSS_VALIDATION_DATA_SIZE):int((i + 1) * CROSS_VALIDATION_DATA_SIZE), ]])

    test_x = test[:, :data_dim - 1]
    test_y = test[:, data_dim - 1]
    return train_x, train_y, train_cv_x, train_cv_y, test_x, test_y, data_dim


if __name__ == '__main__':
    train_x, train_y, train_cv_x, train_cv_y, test_x, test_y, dimension = readData('messidor_features.csv')
    fld = FLD.FLD()
    knn = KNN.KNN()
    #svm = SVM.SVM()
    helper = Helper.Helper()

    print('---------------------------- Fisher\'s Linear Discriminant ----------------------------')
    # FLD: find the best threshold by doing cross-validation
    err_rate = []
    fld_training_time = [timedelta(), 0]
    fld_testing_time = [timedelta(), 0]
    for threshold in frange(0.001, 0.004, 0.000001):
        start = datetime.now()  # .microsecond
        er = helper.crossValidation(train_cv_x, train_cv_y, CROSS_VALIDATION_DATA_GROUP, threshold, fld)
        err_rate.append([threshold, er])
        fld_training_time[0] += getTimeDifference(start)
        fld_training_time[1] += 1
    helper.plotErrorRateCurve(err_rate, 'LDA: Threshold vs Error Rate', 'Threshold')
    threshold = helper.getOptimalParameter(err_rate)

    # do the prediction based on the best threshold
    print('FLD Error Rate: ' + str(fld.predictionErrorRate(test_x, test_y, threshold)))

    # plot the confusion matrix
    helper.plotConfusionMatrix(test_y, fld.predictionResults(test_x, threshold), 'Confusion Matrix of FLD')

    # prepare the ROC curve prediction arrays
    predictions = []
    for threshold in frange(0.0001, 0.1, 0.000001):
        start = datetime.now()  # .microsecond
        predictions.append(fld.predictionResults(test_x, threshold))
        fld_testing_time[0] += getTimeDifference(start)
        fld_testing_time[1] += 1

    # plot ROC curve for FLD
    helper.plotRocCurve(test_y, predictions, 'ROC curve for FLD')
    print('FLD training time (each iteration): ' + str(fld_training_time[0] / fld_training_time[1]))
    print('FLD testing time (each iteration): ' + str(fld_testing_time[0] / fld_testing_time[1]))

    print('')
    print('---------------------------- K Nearest Neighbors ----------------------------')
    # KNN: find the best k by doing cross-validation
    err_rate = []
    knn_training_time = [timedelta(), 0]
    knn_testing_time = [timedelta(), 0]
    for k in range(1, 100):
        start = datetime.now()  # .microsecond
        er = helper.crossValidation(train_cv_x, train_cv_y, CROSS_VALIDATION_DATA_GROUP, k, knn)
        err_rate.append([k, er])
        knn_training_time[0] += getTimeDifference(start)
        knn_training_time[1] += 1
    helper.plotErrorRateCurve(err_rate, 'KNN: K vs Error Rate', 'K')
    k = helper.getOptimalParameter(err_rate)

    # do the prediction based on the best threshold
    print('KNN Error Rate:' + str(knn.predictionErrorRate(test_x, test_y, k)))

    # plot the confusion matrix
    helper.plotConfusionMatrix(test_y, knn.predictionResults(test_x, k), 'Confusion Matrix of KNN')

    # prepare the ROC curve prediction arrays
    predictions = []
    for k in range(1, 100):
        start = datetime.now()  # .microsecond
        predictions.append(knn.predictionResults(test_x, k))
        knn_testing_time[0] += getTimeDifference(start)
        knn_testing_time[1] += 1

    # plot ROC curve for FLD
    helper.plotRocCurve(test_y, predictions, 'ROC curve for KNN')
    print('KNN training time (each iteration): ' + str(knn_training_time[0] / knn_training_time[1]))
    print('KNN testing time (each iteration): ' + str(knn_testing_time[0] / knn_testing_time[1]))

#    print('')
#    print('---------------------------- Support Vector Machine (with Kernel function rbf) ----------------------------')
#    # SVM: find the best k1 by doing cross-validation
#    err_rate = []
#    svm_training_time = [timedelta(), 0]
#    svm_testing_time = [timedelta(), 0]
#    for k1 in frange(1.2, 1.4, 0.02):
#        print('k1 = ' + str(k1))
#        start = datetime.now()  # .microsecond
#        er = helper.crossValidation(train_cv_x, train_cv_y, CROSS_VALIDATION_DATA_GROUP, k1, svm)
#        err_rate.append([k1, er])
#        svm_training_time[0] += getTimeDifference(start)
#        svm_training_time[1] += 1
#    helper.plotErrorRateCurve(err_rate, 'SVM: Sigma vs Error Rate', 'Sigma')
#    k1 = helper.getOptimalParameter(err_rate)
#    svm.training(train_x, train_y, k1)
#    print('')
#
#    # do the prediction based on the best threshold
#    print('SVM Error Rate at'+str(k1)+': ' + str(svm.predictionErrorRate(test_x, test_y, k1)))
#
#    # plot the confusion matrix
#    helper.plotConfusionMatrix(test_y, svm.predictionResults(test_x, k1), 'Confusion Matrix of SVM')
#
#    # prepare the ROC curve prediction arrays
#    predictions = []
#    for k1 in frange(1.2, 1.4, 0.02):
#        start = datetime.now()  # .microsecond
#        predictions.append(svm.predictionResults(test_x, k1))
#        svm_testing_time[0] += getTimeDifference(start)
#        svm_testing_time[1] += 1
#
#    # plot ROC curve for FLD
#    helper.plotRocCurve(test_y, predictions, 'ROC curve for SVM')
#    print('SVM training time (each iteration): ' + str(svm_training_time[0] / svm_training_time[1]))
#    print('SVM testing time (each iteration): ' + str(svm_testing_time[0] / svm_testing_time[1]))

    print('done')
