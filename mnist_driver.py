from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json

from helpers import percentage
from nn import NeuralNetwork, ActivationType
from timer import Timer

from datetime import datetime
from time import time


def plot_digits(X_train, y_train):
    f, axarr = plt.subplots(4, 3)
    axarr = axarr.reshape(1, -1)
    for d in range(50):
        digit, y = X_train[d], y_train[d]
        digit_image = digit.reshape(28, 28)
        axarr[0][y].imshow(digit_image, cmap=mpl.cm.binary, interpolation='nearest')
        axarr[0][y].axis('off')

    axarr[0][10].axis('off')
    axarr[0][11].axis('off')

    plt.axis('off')
    plt.savefig('./digits.png')
    plt.show()


fname = 'logs/assgn/mnist{}.log'.format(str(datetime.now()).replace(':', '.'))
start_time = time()


def append_params(**params):
    with open(fname, 'a') as outf:
        print(json.dumps(params))
        outf.write(json.dumps(params))
        outf.write('\n')


def append_progress(iter, acc):
    with open(fname, 'a') as outf:
        outf.write('{},{},{}\n'.format(iter, acc, time() - start_time))


mnist = input_data.read_data_sets('/tmp/data/')

X_train = mnist.train.images
y_train = mnist.train.labels.astype('int')

X_test = mnist.test.images
y_test = mnist.test.labels.astype('int')

print('X_train.shape={}, y_train.shape={}'.format(X_train.shape, y_train.shape))
print('X_test.shape={}, y_test.shape={}'.format(X_test.shape, y_test.shape))
print('X_train={}'.format(len(X_train)))
print('X_test={}'.format(len(X_test)))

iterations = 5000000000
neurons = [300]
momentum = .0
learning_rate = .5
batch_size = 1000
report_every = 10
best_accuracy = 0
append_params(neurons=neurons, momentum=momentum, learning_rate=learning_rate, batch_size=batch_size)

nn = NeuralNetwork(28 * 28, neurons, [ActivationType.relu, ActivationType.relu], 10, momentum=momentum, learning_rate=learning_rate)
for i in range(iterations):
    if i % report_every == 0:
        with Timer(lambda t: print('Testing took {:.2f} seconds'.format(t))):
            accuracy = nn.test(X_test, y_test)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            append_progress(i, accuracy)
            print('Accuracy: {}, Best Accuracy: {}'.format(percentage(accuracy), percentage(best_accuracy)), end='')
    with Timer(lambda t: print('Training batch {}/{} took {:.2f} seconds'.format((i % 10) + 1, report_every, t))):
        nn.train(X_train, y_train, batch_size=batch_size)
