#!/usr/bin/env python
"""
A stochastic back-propegation neural network is demonstrated.
Implementation is of matrix structure, using the numpy library.
"""

import numpy as np
from numpy import random


# init weights randomly
hiddenWeights = np.array([[random.rand(), random.rand()], [random.rand(), random.rand()]])
outWeights = np.array([[random.rand()], [random.rand()]])
# init biasis randomly
hiddenBias = random.rand()
outBias = random.rand()
# learning rate
learningRate = 0.5
# data set
truthTable = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
])


def main():
    test(truthTable)
    # train for 90 i4terations
    train(90, truthTable)
    # print out the test results
    test(truthTable)


def unk(input):
    # returns the estimate of the nn
    return prop(input)[2]


def train(iterations, dataset):
    # train the nn for iterations on given dataset
    for i in range(iterations):
        # choose a random set from given dataset
        set_num = random.randint(len(dataset))
        # take inputs from dataset
        set = dataset[set_num][:2]
        # take output
        solution = dataset[set_num][-1]
        print(str(set) + " -> " + str(solution) + " expected. recived " + str(prop(set)[2]))
        # run through the nn, and afterwards back propagate with the output
        back_prop(prop(set), solution)


def test(dataset):
    # print all test cases for given datasets and the nn's estimations
    print("printing test results of dataset.")
    for set in dataset:
        print(str(set) + " -> " + str(prop(set[:2])[2]))


def prop(input):
    # propagate
    hidden_values = sig(input.dot(hiddenWeights) + hiddenBias)
    output = sig(hidden_values.dot(outWeights) + outBias)
    # return the whole nn
    return [input, hidden_values, output]


def back_prop(alpha, y):
    # do a backpropagation step
    global hiddenWeights, outWeights, hiddenBias, outBias
    # cost
    delta = alpha[2] - y
    # calculate gradient, and descend.
    hiddenWeights = hiddenWeights - (learningRate * delta) * (alpha[0].dot(outWeights))
    hiddenBias = hiddenBias + (learningRate * delta)
    outWeights = outWeights - (learningRate * delta) * np.reshape(alpha[1], (len(alpha[1]), 1))
    outBias = outBias + (learningRate * delta) 


def sig(x):
    # sigmoid function
    return 1/(1 + np.exp(-x))


if __name__ == "__main__":
    main()
