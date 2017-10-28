import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from helpers import percentage
from nn import NeuralNetwork, ActivationType
from timer import Timer

from datetime import datetime
from time import time

from gendat import gendat
import json

fname = 'logs/assgn/num{}.log'.format(str(datetime.now()).replace(':', '.'))
start_time = time()


def append_params(**params):
    with open(fname, 'a') as outf:
        outf.write(json.dumps(params))
        outf.write('\n')


def append_progress(iter, acc):
    with open(fname, 'a') as outf:
        outf.write('{},{},{}\n'.format(iter, acc, time() - start_time))


def double_shuffle(l1, l2):
    combined = list(zip(l1, l2))
    random.shuffle(combined)
    l1[:], l2[:] = zip(*combined)
    return l1, l2


zs, os = 900, 900
zstest, ostest = 100, 100

pre_rand_train_x = gendat(0, zs) + gendat(1, os)
pre_rand_train_y = [0 for _ in range(900)] + [1 for _ in range(os)]

rand_x, rand_y = double_shuffle(pre_rand_train_x, pre_rand_train_y)

X_train = np.matrix(rand_x)
y_train = np.matrix(rand_y)

X_test = np.matrix(gendat(0, zstest) + gendat(1, ostest))
y_test = np.matrix([0 for _ in range(zstest)] + [1 for _ in range(ostest)])

print('X_train.shape={}, y_train.shape={}'.format(X_train.shape, y_train.shape))
print('X_test.shape={}, y_test.shape={}'.format(X_test.shape, y_test.shape))
print('X_train={}'.format(len(X_train)))
print('X_test={}'.format(len(X_test)))

neurons = [5, 10]
activation = [ActivationType.relu, ActivationType.relu]
batch_size = 10
momentum = 0.8
learning_rate = .005
append_params(neurons=neurons, learning_rate=learning_rate, batch_size=batch_size, momentum=momentum)
nn = NeuralNetwork(2, neurons, activation, 2, momentum=momentum, learning_rate=learning_rate)

with Timer(lambda t: print('Took {} seconds'.format(t))):
    accuracy = nn.test(X_train.tolist(), y_train.tolist()[0])
    print('Accuracy: {}'.format(percentage(accuracy)))

iterations = 100000
for i in range(iterations):
    if i % 1 == 0 or i == 0:
        print('Testing...')
        with Timer(lambda t: print('Testing took {} seconds'.format(t))):
            accuracy = nn.test(X_test.tolist(), y_test.tolist()[0])
            append_progress(i, accuracy)
            print('Accuracy: {}'.format(percentage(accuracy)))
    with Timer(lambda t: print('Training took {} seconds'.format(t))):
        print('Training...')
        nn.train(X_train.tolist(), y_train.tolist()[0], batch_size=batch_size)
