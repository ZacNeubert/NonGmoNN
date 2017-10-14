from random import random
import numpy as np


class OutputType:
    softmax = lambda outputs: [o / sum(outputs) for o in outputs]
    sigmoid = 1


class ActivationType:
    relu = lambda out: max(0, out)
    sum = lambda out: sum([out])
    sigmoid = 1


class Neuron:
    def __init__(self, input_size, activation_type, learning_rate):
        self.input_size = input_size
        self.activation_type = activation_type
        self.learning_rate = learning_rate
        self.weights = np.matrix([random() for _ in range(input_size)])
        self.bias = random()

    def __str__(self):
        return '''
        Neuron:
        Weights: {}
        {}
        '''.format(
            len(self.weights),
            ','.join(['{:.02f}'.format(w) for w in self.weights])
        )

    def __repr__(self):
        return str(self)

    def get_output(self, inputs):
        out = self.weights.dot(inputs)
        out = self.activation_type(self.bias + out)
        return out[0,0]

    def train(self, inputs, expected_output):
        output = self.get_output(inputs)
        difference = output - expected_output
        self.adjust_weights(difference)

    def adjust_weights(self, difference):
        pass


class Layer:
    def __init__(self, input_size, neuron_count, learning_rate, activation_type=ActivationType.relu):
        self.input_size = input_size
        self.neuron_count = neuron_count
        self.neurons = [Neuron(input_size=input_size,
                               activation_type=activation_type,
                               learning_rate=learning_rate) for _ in range(neuron_count)]

    def process_input(self, inputs):
        output = [neuron.get_output(inputs=inputs) for neuron in self.neurons]
        output = np.matrix(output)
        return output

    def __str__(self):
        return '''
    Layer:
    Neurons: {}
    {}
        '''.format(
            len(self.neurons),
            '\n'.join([str(neuron) for neuron in self.neurons])
        )

    def __repr__(self):
        return str(self)


class NeuralNetwork:
    def __init__(self, input_count, hidden_layer_neuron_counts, hidden_layer_activations, output_count,
                 output_type=OutputType.softmax, momentum=0, learning_rate=.05):
        self.input_count = input_count
        self.hidden_layer_neuron_counts = hidden_layer_neuron_counts
        self.hidden_layer_activations = hidden_layer_activations
        self.output_count = output_count
        self.output_type = output_type
        self.momentum = momentum
        self.learning_rate = learning_rate

        # each layer's input size is the number of neurons in the previous layer
        self.layers = []
        for i, (count, activation) in enumerate(zip(self.hidden_layer_neuron_counts, self.hidden_layer_activations)):
            print('Appended layer')
            if i == 0:
                self.layers.append(Layer(input_size=self.input_count,
                                         neuron_count=count,
                                         activation_type=activation,
                                         learning_rate=self.learning_rate))
            else:
                self.layers.append(Layer(input_size=self.layers[i - 1].neuron_count,
                                         neuron_count=count,
                                         activation_type=activation,
                                         learning_rate=self.learning_rate))

        self.layers.append(Layer(input_size=self.layers[-1].neuron_count,
                                 neuron_count=self.output_count,
                                 activation_type=ActivationType.sum,
                                 learning_rate=self.learning_rate))

    def __str__(self):
        return '''
NN:
Layers: {}
{}
        '''.format(
            len(self.layers),
            '\n'.join([str(layer) for layer in self.layers])
        )

    def __repr__(self):
        return str(self)

    def train(self, data):
        pass

    def test(self, test_data, test_answers, report_every=5, report_success=False):
        correct = 0
        total = 0
        for datum, true_answer in zip(test_data, test_answers):
            total += 1
            if total % report_every == 0:
                print('Classified {} out of {}'.format(total, len(test_answers)))
            answer = self.classify(datum)
            if report_success and answer == true_answer:
                print('CORRECT')
                correct += 1

        return correct / total

    def classify(self, datum):
        inputs = datum
        for layer in self.layers:
            inputs = layer.process_input(inputs.reshape(-1, 1))
        output = self.output_type(inputs)
        return output.index(max(output))
