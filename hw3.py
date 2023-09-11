import numpy as np
import matplotlib.pyplot as plt

def loadData(filePath, seed=None):
    if seed is None:
        np.random.seed(0)
    else:
        np.random.seed(seed)
    data = np.genfromtxt(filePath, delimiter=',')
    np.random.shuffle(data)
    seperate = np.array_split(data, 3)
    return np.concatenate((seperate[0], seperate[1])), seperate[2]

def getXY(data):
    X = data[:len(data), :len(data[0])-1]
    Y = data[:len(data), len(data[0])-1:]
    return X, Y

def standardize(d, valid):
    data, classifier = getXY(d)
    validData, validClassifier = getXY(valid)
    trainZscore = (data-np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)
    validZscore = (validData-np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)
    return np.concatenate((trainZscore, classifier), axis=1), np.concatenate((validZscore, validClassifier), axis=1)

def getUniqueBinaryClasses(d):
    return d[np.isin(element=d[:len(d), len(d[0])-1], test_elements=[0])], d[np.isin(element=d[:len(d), len(d[0])-1], test_elements=[1])]

def mean(d):
    class0, class1 = getUniqueBinaryClasses(d)
    return np.mean(class0[:, :-1], axis=0), np.mean(class1[:, :-1], axis=0)

def covariance(d):
    class0, class1 = getUniqueBinaryClasses(d)
    return np.cov(class0[:, :-1], rowvar=False), np.cov(class1[:, :-1], rowvar=False)
    
def eigan(d):
    return np.linalg.eig(d)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logLoss(yTrue, yHat):
    # all zero values become very small values in case of log(0)
    yHat = np.clip(yHat, 1e-10, 1 - 1e-10)
    return -np.mean(yTrue*np.log(yHat) + (1-yTrue)*np.log(1-yHat))

def part2():
    dataPath = "spambase.data"

    train, validation = loadData(dataPath)
    trainStandard, validStandard = standardize(train, validation)
    mean0, mean1 = mean(trainStandard)
    sigma0, sigma1 = covariance(trainStandard)

    sb = np.dot(np.array([np.subtract(mean0, mean1)]).transpose(), np.array([np.subtract(mean0, mean1)]))
    sw = np.add(sigma0, sigma1)

    eiganValues, eiganVectors = eigan(np.dot(np.linalg.inv(sw), sb))
    nonZero = np.argmax(eiganValues)
    w = eiganVectors[: ,nonZero]

    validationPredictions = []
    trainPredictions = []
    validationProjection = np.dot(validStandard[:, :-1], w)
    trainProjection = np.dot(trainStandard[:, :-1], w)
    # classify data
    for i in range(0, validation.shape[0]):
        distance0 = np.linalg.norm(validationProjection[i] - np.dot(mean0, w))
        distance1 = np.linalg.norm(validationProjection[i] - np.dot(mean1, w))
        if distance0 < distance1:
            validationPredictions.append(0)
        else:
            validationPredictions.append(1)
    for i in range(0, train.shape[0]):
        distance0 = np.linalg.norm(trainProjection[i] - np.dot(mean0, w))
        distance1 = np.linalg.norm(trainProjection[i] - np.dot(mean1, w))
        if distance0 < distance1:
            trainPredictions.append(0)
        else:
            trainPredictions.append(1)

    trainAccuracy = np.mean(train[:, -1] == np.array(trainPredictions))
    validationAccurary = np.mean(validation[:, -1] == np.array(validationPredictions))
    tp = np.sum((validation[:, -1] == 1) & (np.array(validationPredictions) == 1))
    fp = np.sum((validation[:, -1] == 0) & (np.array(validationPredictions) == 1))
    fn = np.sum((validation[:, -1] == 1) & (np.array(validationPredictions) == 0))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = 2 * precision * recall / (precision + recall)

    print("LDA")
    print("Training Accuracy:", trainAccuracy)
    print("Validation Accuracy:", validationAccurary)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-Measure:", fmeasure)

def part3():
    dataPath = "spambase.data"

    train, validation = loadData(dataPath)
    # classifier included with added bias feature
    trainStandard, validStandard = standardize(train, validation)
    trainStandard, validStandard = np.hstack((np.ones((len(train), 1)), trainStandard)), np.hstack((np.ones((len(validation), 1)), validStandard))

    # w size is without classifier but with bias
    w = np.random.rand(trainStandard.shape[1]-1, 1)
    l = 0.01
    epochs = 10000
    N = train.shape[0]

    trainLoss = []
    validationLoss = []
    # get data and target values for training and validation data
    Xtrain, Ytrain = getXY(trainStandard)
    Xvalid, Yvalid = getXY(validStandard)
    for _ in range(0, epochs):
        # compute the sigmoid for both sets of data
        trainZ, validZ = np.dot(Xtrain, w), np.dot(Xvalid, w)
        trainYhat, validYhat = sigmoid(trainZ), sigmoid(validZ)
        # compute the log-loss for the sets of data
        trainLogLoss = logLoss(Ytrain, trainYhat)
        validLogLoss = logLoss(Yvalid, validYhat)
        # compute the new weights based on the updated gradient
        trainGradient = (1/N) * np.dot(Xtrain.transpose(), np.subtract(trainYhat, Ytrain))
        w -= l*trainGradient
        # append the log loss values
        trainLoss.append(trainLogLoss)
        validationLoss.append(validLogLoss)
    
    validZ = np.dot(Xvalid, w)
    yHat = sigmoid(validZ)
    predictions = np.where(yHat >= 0.5, 1, 0)

    tp = np.sum((predictions == 1) & (Yvalid == 1))
    fp = np.sum((predictions == 1) & (Yvalid == 0))
    fn = np.sum((predictions == 0) & (Yvalid == 1))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = 2 * precision * recall / (precision + recall)
    validationAccurary = np.mean(Yvalid == predictions)

    print("Logistic Regression")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-Measure:", fmeasure)
    print("Accuracy:", validationAccurary)

    # blue line
    plt.plot(trainLoss, label="Training loss")
    # orange line
    plt.plot(validationLoss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.show()
     
if __name__ == "__main__":
    part2()
    print("#~~~~~~~~~~~~~~~~~~~~~~~#")
    part3()