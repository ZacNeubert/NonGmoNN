import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

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


def read_num(i):
    numfilename = 'num{}.csv'.format(i)
    with open(numfilename, 'r') as infile:
        lines = [[float(f) for f in l.split(',')] for l in infile.readlines()]
    return lines


zeros, ones = 900, 900
zstest, ostest = 100, 100

num0 = read_num(0)
num1 = read_num(1)
pre_rand_train_x = num0 + num1
pre_rand_train_y = [0 for _ in range(len(num0))] + [1 for _ in range(len(num1))]

rand_x, rand_y = double_shuffle(pre_rand_train_x, pre_rand_train_y)

X_train = np.matrix(rand_x)
y_train = np.matrix(rand_y)

X_test = np.matrix(gendat(0, zstest) + gendat(1, ostest))
y_test = np.matrix([0 for _ in range(zstest)] + [1 for _ in range(ostest)])

print('X_train.shape={}, y_train.shape={}'.format(X_train.shape, y_train.shape))
print('X_test.shape={}, y_test.shape={}'.format(X_test.shape, y_test.shape))
print('X_train={}'.format(len(X_train)))
print('X_test={}'.format(len(X_test)))

neurons = [5]
activation = [ActivationType.relu, ActivationType.relu]
momentum = 0.0
learning_rate = .000005
iterations = 150
append_params(neurons=neurons, learning_rate=learning_rate, momentum=momentum)
nn = NeuralNetwork(2, neurons, activation, 2, momentum=momentum, learning_rate=learning_rate)

with Timer(lambda t: print('Took {} seconds'.format(t))):
    accuracy = nn.test(X_train.tolist(), y_train.tolist()[0])
    print('Accuracy: {}'.format(percentage(accuracy)))

mean_mses = []
for i in range(iterations):
    if i % 10 == 0 or i == 0:
        accuracy = nn.test(X_test.tolist(), y_test.tolist()[0])
        append_progress(i, accuracy)
        print('{}/{} Accuracy: {}'.format(i, iterations, percentage(accuracy)))
    mean_mse = nn.train(X_train.tolist(), y_train.tolist()[0], shush=True)
    print(mean_mse)
    mean_mses.append(mean_mse)

plt.plot(mean_mses)
plt.suptitle('Mean MSE over time B={}'.format(momentum))
figname = './mseb{}.png'.format(int(momentum * 10))
plt.savefig(figname)
print(mean_mses)
