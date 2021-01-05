#!/usr/bin/env python
"""
A batch back-propegation neural network is demonstrated.
Implementation is of matrix structure, using the numpy library.
"""

import numpy as np
from numpy import random


# init weights randomly
hiddenWeights = np.array([[random.rand(), random.rand()], [random.rand(), random.rand()]])
outWeights = np.array([[random.rand()], [random.rand()]])
# init biases randomly
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
    # train for 50 batches, of 4 iterations each
    train(4, 50, truthTable)
    # print out the test results
    test(truthTable)


def unk(input):
    # returns the estimate of the nn
    return prop(input)[2]


def train(iterations_per_batch, batches, dataset):
    # train the nn for iterations on given dataset
    for batch_num in range(batches):
        batch_sum = [0, 0, 0, 0]
        for iteration in range(iterations_per_batch):
            # choose a random set from given dataset
            set_num = random.randint(len(dataset))
            # take inputs from dataset
            set = dataset[set_num][:2]
            # take output
            solution = dataset[set_num][-1]
            print(str(set) + " -> " + str(solution) + " expected. received " + str(prop(set)[2]))
            # in batch back-prop, we sum all the changes we need (this is called an epoch). we apply them all at once
            batch_sum = [x + y for x, y in zip(back_prop(prop(set), solution), batch_sum)]
        global hiddenWeights, outWeights, hiddenBias, outBias
        # apply all the changes
        hiddenWeights = hiddenWeights - batch_sum[0]
        outWeights = outWeights - batch_sum[1]
        hiddenBias = hiddenBias + batch_sum[2]
        outBias = outBias + batch_sum[2]
        print("finished batch " + str(batch_num + 1) + " (epoch " + str(batch_num) + ")")


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
    delta = (alpha[2] - y) * learningRate
    # calculate gradient
    hidden_weights_diff = delta * (alpha[0].dot(outWeights))
    out_weights_diff = delta * np.reshape(alpha[1], (len(alpha[1]), 1))
    return [hidden_weights_diff, out_weights_diff, delta]


def sig(x):
    # sigmoid function
    return 1/(1 + np.exp(-x))


if __name__ == "__main__":
    main()
