import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import Helper
import FLD
import KNN
import NB


TRAINNING_DATA_SIZE = 840  # make sure that this is divisible by CV_DATA_GROUP
CROSS_VALIDATION_DATA_GROUP = 4
CROSS_VALIDATION_DATA_SIZE = (TRAINNING_DATA_SIZE / CROSS_VALIDATION_DATA_GROUP)
MAX_MICROSECOND = 99999


def frange(x, y, jump):
    while x + jump <= y:
        yield x
        x += jump

def expRange(x, y, base, step):
    while x + step <= y:
        yield pow(base, x)
        x += step

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
    nb = NB.NB()
    helper = Helper.Helper()

    print('---------------------------- Fisher\'s Linear Discriminant ----------------------------')
    # FLD: find the best threshold by doing cross-validation
    err_rate = []
    fld_cv_training_time = [timedelta(), 0]
    fld_training_time = timedelta()
    fld_testing_time = timedelta()
    for threshold in frange(0.001, 0.004, 0.000001):
        start = datetime.now()
        er = helper.crossValidation(train_cv_x, train_cv_y, CROSS_VALIDATION_DATA_GROUP, threshold, fld)
        err_rate.append([threshold, er])
        fld_cv_training_time[0] += getTimeDifference(start)
        fld_cv_training_time[1] += 1
    helper.plotErrorRateCurve(err_rate, 'Fisher\' Linear Discriminant: Threshold vs Error Rate', 'Threshold')
    threshold = helper.getOptimalParameter(err_rate)

    #training using the whole set of training data
    start = datetime.now()
    fld.training(train_x, train_y, threshold)
    fld_training_time += getTimeDifference(start)

    # do the prediction based on the best threshold
    print('Fisher\' Linear Discriminant (FLD) Error Rate(threshold='+str(threshold)+'): ' + str(fld.predictionErrorRate(test_x, test_y, threshold)))

    # prepare the prediction arrays for ROC curve and Confusion matrix
    start = datetime.now()
    fldPrediction = fld.predictionResults(test_x, threshold)
    fld_testing_time += getTimeDifference(start)

    print('FLD cross validation training time (each iteration): ' + str(fld_cv_training_time[0] / fld_cv_training_time[1]))
    print('FLD training time: ' + str(fld_training_time))
    print('FLD testing time: ' + str(fld_testing_time))

    print('')
    print('---------------------------- K Nearest Neighbors ----------------------------')
    # KNN: find the best k by doing cross-validation
    err_rate = []
    knn_cv_training_time = [timedelta(), 0]
    knn_training_time = timedelta()
    knn_testing_time = timedelta()
    for k in range(1, 50):
        start = datetime.now()
        er = helper.crossValidation(train_cv_x, train_cv_y, CROSS_VALIDATION_DATA_GROUP, k, knn)
        err_rate.append([k, er])
        knn_cv_training_time[0] += getTimeDifference(start)
        knn_cv_training_time[1] += 1
    helper.plotErrorRateCurve(err_rate, 'K Nearest Neighbors: K vs Error Rate', 'K')
    k = helper.getOptimalParameter(err_rate)

    #training using the whole set of training data
    start = datetime.now()
    knn.training(train_x, train_y, k)
    knn_training_time += getTimeDifference(start)

    # do the prediction based on the best k
    print('K Nearest Neighbors (KNN) Error Rate (k='+str(k)+'): ' + str(knn.predictionErrorRate(test_x, test_y, k)))

    # prepare the ROC curve prediction arrays
    start = datetime.now()
    knnPrediction = knn.predictionResults(test_x, k)
    knn_testing_time += getTimeDifference(start)

    print('KNN cross validation training time (each iteration): ' + str(
        knn_cv_training_time[0] / knn_cv_training_time[1]))
    print('KNN training time: ' + str(knn_training_time))
    print('KNN testing time: ' + str(knn_testing_time))

    print('')
    print('---------------------------- Naive Bayes ----------------------------')
    # Naive Bayes: Since there is no parameter to optimize, the cross-validation and ROC curve part will be omitted
    err_rate = []
    nb_cv_training_time = [timedelta(), 0]
    nb_training_time = timedelta()
    nb_testing_time = timedelta()
    for ratio in expRange(-1, 0.2, 10, 0.01):
        start = datetime.now()
        er = helper.crossValidation(train_cv_x, train_cv_y, CROSS_VALIDATION_DATA_GROUP, ratio, nb)
        err_rate.append([ratio, er])
        nb_cv_training_time[0] += getTimeDifference(start)
        nb_cv_training_time[1] += 1
    helper.plotErrorRateCurve(err_rate, 'Naive Bayes: ratio vs Error Rate', 'ratio')
    ratio = helper.getOptimalParameter(err_rate)

    # training using the whole set of training data
    start = datetime.now()
    nb.training(train_x, train_y, ratio)
    nb_training_time += getTimeDifference(start)

    # do the prediction based on the best ratio
    print('Naive Bayes (NB) Error Rate (ratio='+str(ratio)+'): ' + str(nb.predictionErrorRate(test_x, test_y, ratio)))

    # prepare the ROC curve prediction arrays
    start = datetime.now()
    nbPrediction = nb.predictionResults(test_x, ratio)
    nb_testing_time += getTimeDifference(start)

    print('NB cross validation training time (each iteration): ' + str(
        nb_cv_training_time[0] / nb_cv_training_time[1]))
    print('NB training time: ' + str(nb_training_time))
    print('NB testing time: ' + str(nb_testing_time))

    # ---------------------------- print out all the confusion matrices and ROC curve ----------------------------
    # plot the confusion matrix
    helper.plotConfusionMatrix(test_y, fldPrediction, 'Confusion Matrix of FLD')
    # plot the confusion matrix
    helper.plotConfusionMatrix(test_y, knnPrediction, 'Confusion Matrix of KNN')
    # plot the confusion matrix
    helper.plotConfusionMatrix(test_y, nbPrediction, 'Confusion Matrix of NB')

    helper.newFigure()
    # plot ROC curves
    helper.plotRocCurve(test_y, fldPrediction, 'Fisher\' Linear Discriminant', 'g')
    # plot ROC curve for FLD
    helper.plotRocCurve(test_y, knnPrediction, 'K Nearest Neighbour', 'b')
    # plot ROC curve for FLD
    helper.plotRocCurve(test_y, nbPrediction, 'Naive Bayes', 'y')

    print('done')
