import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        #flatten images into 2d arrays
        Data = Data.transpose(2, 0, 1)
        Data = Data.reshape(3745,784)
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget



def MSE(W, b, x, y, reg):
    error = tf.matmul(x, W) + b - y
    regularize = reg * tf.math.square(W)/2
    mse_loss = (tf.reduce_sum(tf.square(error))/(2*y.size)) + regularize
    return mse_loss

def gradMSE(W, b, x, y, reg):
    error = tf.matmul(x, W) + b - y
    gradient  = tf.matmul(tf.transpose(x),error)/y.size + reg * W
    return gradient

def grad_descent(weight, b, xtrain, ytrain, learning_rate, epochs, reg, error_tol):
    #alpha is learning_rate; epochs is # iteration t
    #initialize x, y and weights
    xtrain = tf.pad(xtrain,[[1,0],[0,0]], "CONSTANT", constant_values = 1)
    ytrain = tf.cast(tf.constant(ytrain, shape=[np.shape(ytrain)[0] ,1]), dtype=tf.float64)
    bias = 1
    reg = 1
    error = 1

    for i in range(0, epochs) and error > error_tol:
        gradient = gradMSE(weight, bias, xtrain, ytrain, reg)
        if gradient == 0:
            break
        elif gradient > 0:
            prev_weight = weight
            weight = weight - learning_rate * tf.reduce_sum(gradient, 1, True) / gradient.size
        else:
            weight = weight + learning_rate * tf.reduce_sum(gradient, 1, True) / gradient.size
        error = tf.matmul((weight - prev_weight),tf.transpose(weight - prev_weight))
    return weight, error


def tuning_the_learning_rate():
    # set up parameters
    epochs = 5000
    reg = 0
    learning_rates = [0.005, 0.001, 0.0001]
    error_tol = 10**-3

    (trainData, trainTarget,
     testData, testTarget,
     validData, validTarget) = loadData()
    weight = tf.constant(0., shape=[np.shape(trainData)[0]+1, 1], dtype=tf.float64)


    results = []

    for i in range(0, len(learning_rates)):
        results.append(
        grad_descent(weight, trainData, trainTarget, learning_rates[i], epochs, reg, error_tol) + (learning_rates[i],))

    return results

