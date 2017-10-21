from random import random, shuffle
import numpy as np
from math import e, exp


def prepend_1(matrix):
    if isinstance(matrix, list):
        l = matrix
    else:
        l = matrix.tolist()
    if isinstance(l[0], list):
        l = l[0]
    return np.matrix([1] + l)


class OutputFunction:
    def __init__(self, func, deriv):
        self.func = func
        self.deriv = deriv

    def __call__(self, *args, deriv=False, **kwargs):
        largest = max(args[0])
        smallest = min(args[0])
        try:
            if deriv:
                return self.deriv(*args, **kwargs)
            else:
                return self.func(*args, **kwargs)
        except Exception as e:
            print(e)
            return None


def softmax_func(outputs):
    m = max(outputs)
    return [exp(o - m) / sum([exp(out - m) for out in outputs]) for o in outputs]


class OutputType:
    softmax = OutputFunction(
        func=softmax_func,
        deriv=lambda outputs, n: exp(outputs[n])
    )
    sigmoid = 1


class ActivationType:
    relu = lambda out: max(0, out)
    sum = lambda out: sum([out])
    sigmoid = 1


class Neuron:
    def __init__(self, input_size, activation_type, learning_rate, beta=0.8):
        self.input_size = input_size
        self.activation_type = activation_type
        self.learning_rate = learning_rate
        self.weights = np.matrix([random() for _ in range(input_size)])
        self.beta = beta
        self.last_output = None
        self.last_inputs = None
        self.last_changes = None

    def __str__(self):
        return '''
        Neuron:
        Weights: {}
        {}
        '''.format(
            len(self.weights),
            ','.join(['{:.02f}'.format(w) for w in self.weights.tolist()[0]])
        )

    def __repr__(self):
        return str(self)

    def get_weight(self, n):
        return self.weights[0, n]

    def get_output(self, inputs):
        out = self.weights.dot(inputs)
        self.last_inputs = inputs
        self.last_output = out[0, 0]
        out = self.activation_type(out)
        if isinstance(out, np.matrix):
            return out[0, 0]
        elif isinstance(out, int):
            return out
        else:
            print(type(out))
            raise Exception('Shit happened')

    def train(self, inputs, expected_output):
        output = self.get_output(inputs)
        difference = output - expected_output
        self.adjust_weights(difference)

    def adjust_weights(self, difference):
        changes = [self.learning_rate * x * -difference
                   for x, w in zip(self.last_inputs.reshape(-1).tolist()[0],
                                   self.weights.tolist()[0])]
        if self.last_changes and self.beta:
            changes = [c * self.beta + old_c * (1 - self.beta) for c, old_c in zip(changes, self.last_changes)]
        self.last_changes = changes
        self.weights = np.matrix([w + c for c, w in zip(changes, self.weights.tolist()[0])])


class Layer:
    def __init__(self, input_size, neuron_count, learning_rate, activation_type=ActivationType.relu, momentum=0.0):
        self.iter = 0
        self.input_size = input_size
        self.momentum = momentum
        self.neuron_count = neuron_count
        self.neurons = [Neuron(input_size=input_size,
                               activation_type=activation_type,
                               beta=momentum,
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
                 output_type=OutputType.softmax, momentum=0, learning_rate=.005):
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
            if i == 0:
                self.layers.append(Layer(input_size=self.input_count + 1,
                                         neuron_count=count,
                                         activation_type=activation,
                                         momentum=self.momentum,
                                         learning_rate=self.learning_rate))
            else:
                self.layers.append(Layer(input_size=self.layers[i - 1].neuron_count + 1,
                                         neuron_count=count,
                                         activation_type=activation,
                                         momentum=self.momentum,
                                         learning_rate=self.learning_rate))

        self.layers.append(Layer(input_size=self.layers[-1].neuron_count + 1,
                                 neuron_count=self.output_count,
                                 activation_type=ActivationType.sum,
                                 momentum=self.momentum,
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

    @staticmethod
    def cost(output, correct_output):
        return [.5 * (o - c) ** 2 for o, c in zip(output, correct_output)]

    def train(self, data, answers, batch_size=0):
        shufflable = list(zip(data, answers))
        shuffle(shufflable)
        count = 0
        for datum, answer in shufflable:
            count += 1
            if batch_size:
                if count > batch_size:
                    return
            correct = [1 if i == answer else 0 for i in range(self.output_count)]
            outputs, outputs_by_layer = self.process_input(datum)
            # costs = self.cost(outputs, correct)
            diffs = outputs - np.matrix(correct)
            calc_answer = outputs.index(max(outputs))
            if answer != calc_answer:
                for diff, neuron in zip(diffs.tolist()[0], self.layers[-1].neurons):
                    neuron.adjust_weights(diff)
                    for n, middle_neuron in enumerate(self.layers[-2].neurons):
                        middle_diff = diff * neuron.get_weight(n)
                        middle_neuron.adjust_weights(middle_diff)
                        if len(self.layers) > 2:
                            for m, back_neuron in enumerate(self.layers[-3].neurons):
                                back_diff = middle_diff * middle_neuron.get_weight(m)
                                back_neuron.adjust_weights(back_diff)

    def test(self, test_data, test_answers, report=False, report_every=5, report_success=False):
        correct = 0
        total = 0
        for datum, true_answer in zip(test_data, test_answers):
            total += 1
            if total % report_every == 0 and report:
                print('Classified {} out of {}'.format(total, len(test_answers)))
            answer = self.classify(datum)
            if answer == true_answer:
                if report_success:
                    print('CORRECT c{}==t{}'.format(answer, true_answer))
                correct += 1

        return correct / total

    def process_input(self, datum):
        datum = prepend_1(datum)
        outputs_by_layer = [datum]
        for layer in self.layers:
            datum = prepend_1(layer.process_input(datum.reshape(-1, 1)))
            outputs_by_layer.append(datum)
        outputs = self.output_type(datum.tolist()[0][1:])
        outputs_by_layer.append(outputs)
        return outputs, outputs_by_layer

    def classify(self, datum):
        output, outputs_by_layer = self.process_input(datum)
        answer = output.index(max(output))
        return answer
