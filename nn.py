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
        try:
            if deriv:
                return self.deriv(*args, **kwargs)
            else:
                return self.func(*args, **kwargs)
        except Exception as e:
            print(e)
            return None


cache = {}
def cached_exp(n):
    global cache
    if n in cache.keys():
        return cache[n]
    else:
        if len(cache) > 1000:
            cache = {}
        result = exp(n)
        cache[n] = result
        return result


def softmax_func(outputs):
    m = max(outputs)
    return [cached_exp(o - m) / sum([cached_exp(out - m) for out in outputs]) for o in outputs]


def softmax_single(i, x_vec):
    m = max(x_vec)
    x = x_vec[i]
    return cached_exp(x - m) / sum([cached_exp(o - m) for o in x_vec])


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


def estimate_deriv(func, i, *args, **kwargs):
    d = .1
    return (func(i + d, *args, **kwargs) - func(i - d, *args, **kwargs)) / (2 * d)


def softmax_deriv(i, x_vec):
    for d in (1, 10, 100, 1000, 10000, 100000):
        smaller = [x if j != i else x - d for j, x in enumerate(x_vec)]
        larger = [x if j != i else x + d for j, x in enumerate(x_vec)]
        larger_out = softmax_single(i, larger)
        smaller_out = softmax_single(i, smaller)
        result = (larger_out - smaller_out) / (2 * d)
        if result:
            return result
    return result


class Neuron:
    def __init__(self, input_size, activation_type, learning_rate, beta=0.8):
        self.input_size = input_size
        self.activation_type = activation_type
        self.learning_rate = learning_rate
        self.weights = np.matrix([random() for _ in range(input_size)])
        self.beta = beta
        self.pre_act_last_output = None
        self.post_act_last_output = None
        self.last_inputs = None
        self.last_changes = None

    def __str__(self):
        return '''
        Neuron:
        Weights: {}
        {}
        '''.format(
            str(self.weights.shape),
            ','.join(['{:.02f}'.format(w) for w in self.weights.tolist()[0]])
        )

    def __repr__(self):
        return str(self)

    def get_weight(self, n):
        return self.weights[0, n]

    def get_output(self, inputs):
        self.last_inputs = inputs
        out = self.weights.dot(inputs)
        self.pre_act_last_output = out[0, 0]
        out = self.activation_type(out)
        if isinstance(out, np.matrix):
            self.post_act_last_output = out[0, 0]
            return out[0, 0]
        elif isinstance(out, int):
            self.post_act_last_output = out
            return out
        else:
            print(type(out))
            raise Exception('Something bad happened')

    def adjust_weights(self, grad):
        changes = [self.learning_rate * grad * x[0, 0] for x in self.last_inputs]
        if self.last_changes and self.beta:
            changes = [c * self.beta + old_c * (1 - self.beta) for c, old_c in zip(changes, self.last_changes)]
        self.weights = np.matrix([w - c for w, c in zip(self.weights.tolist()[0], changes)])
        self.last_changes = changes


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

    @property
    def layer_weights(self):
        return np.matrix([n.weights.tolist()[0] for n in self.neurons])

    def process_input(self, inputs):
        return inputs
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
            costs = self.cost(outputs, correct)
            diffs = outputs - np.matrix(correct)
            calc_answer = outputs.index(max(outputs))
            if answer != calc_answer:
                for i, (diff, neuron) in enumerate(zip(diffs.tolist()[0], self.layers[-1].neurons)):
                    pre_out = outputs_by_layer[-2].tolist()[0][1:]
                    grad = softmax_deriv(i, x_vec=pre_out)
                    grad = grad * diff
                    neuron.adjust_weights(grad)
                    for n, middle_neuron in enumerate(self.layers[-2].neurons):
                        if middle_neuron.post_act_last_output > 0:
                            middle_diff = grad * neuron.get_weight(n + 1)
                            middle_neuron.adjust_weights(middle_diff)
                            #if len(self.layers) > 2:
                            #    for m, back_neuron in enumerate(self.layers[-3].neurons):
                            #        back_diff = middle_diff * middle_neuron.get_weight(m + 1)
                            #        back_neuron.adjust_weights(back_diff)

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
