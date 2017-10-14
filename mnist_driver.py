from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from nn import NeuralNetwork, ActivationType
from timer import Timer


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


mnist = input_data.read_data_sets('/tmp/data/')

X_train = mnist.train.images
y_train = mnist.train.labels.astype('int')

X_test = mnist.test.images
y_test = mnist.test.labels.astype('int')

print('X_train.shape={}, y_train.shape={}'.format(X_train.shape, y_train.shape))
print('X_test.shape={}, y_test.shape={}'.format(X_test.shape, y_test.shape))
print('X_train={}'.format(len(X_train)))
print('X_test={}'.format(len(X_test)))

#plot_digits(X_train, y_train)

limit = 1000
nn = NeuralNetwork(28*28, [int((28*28 + 10)/2)], [ActivationType.relu], 10)
with Timer(lambda t: print('Took {} seconds'.format(t))):
    accuracy = nn.test(X_train[:limit], y_train[:limit], report_every=100)
    print('Accuracy: {}'.format(accuracy))
