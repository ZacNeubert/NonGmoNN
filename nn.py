from random import random, shuffle
import numpy as np
from math import e, exp
from statistics import mean


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
        if len(cache) > 100:
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
    relu = lambda out: np.matrix([i if i > 0 else 0 for i in vec_to_list(out)])
    sum = lambda out: out
    sigmoid = 1


def vec_to_list(v):
    return v.tolist()[0]


def estimate_deriv(func, i, *args, **kwargs):
    d = .1
    return (func(i + d, *args, **kwargs) - func(i - d, *args, **kwargs)) / (2 * d)


def softmax_deriv(i, x_vec):
    for d in (1, 10, 100, 1000, 10000, 100000):
        smaller = [x if j != i else x - d for j, x in enumerate(vec_to_list(x_vec))]
        larger = [x if j != i else x + d for j, x in enumerate(vec_to_list(x_vec))]
        larger_out = softmax_single(i, larger)
        smaller_out = softmax_single(i, smaller)
        result = (larger_out - smaller_out) / (2 * d)
        if result:
            return result
    return result


class Layer:
    def __init__(self, input_size, neuron_count, learning_rate, activation_type=ActivationType.relu, momentum=0.0):
        self.iter = 0
        self.activation = activation_type
        self.input_size = input_size
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.neuron_count = neuron_count
        self.weights = np.matrix([[random() / 100 for _ in range(input_size)] for __ in range(neuron_count)])
        self.last_inputs = None
        self.post_act_last_output = None
        self.pre_act_last_output = None

    def process_input(self, inputs):
        self.last_inputs = inputs
        self.pre_act_last_output = self.weights.dot(inputs).T
        self.post_act_last_output = self.activation(self.pre_act_last_output)
        return self.post_act_last_output

    def __str__(self):
        return '''
    Layer:
    Shape: {}
        '''.format(
            self.weights.shape,
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
    def mse(output, correct_output):
        return [(o - c) ** 2 for o, c in zip(output, correct_output)]

    def train(self, data, answers, batch_size=0, shush=False):
        shufflable = list(zip(data, answers))
        shuffle(shufflable)
        count = 0
        wrong = 0
        old_back_changes = None
        old_middle_changes = None
        old_output_changes = None
        mses = []
        for datum, answer in shufflable:
            count += 1
            if batch_size:
                if count > batch_size:
                    correct_count = batch_size - wrong
                    if not shush:
                        print('{:.2f}%, {}/{} correct_count on training data'.format(correct_count / batch_size * 100,
                                                                                     correct_count, batch_size))
                    return mean(mses)
            outputs, outputs_by_layer = self.process_input(datum)
            calc_answer = outputs.index(max(outputs))

            y = np.matrix([1 if i == answer else 0 for i in range(self.output_count)])
            y_hat = np.matrix(outputs)

            mses.append(mean(NeuralNetwork.mse(vec_to_list(y_hat), vec_to_list(y))))

            if answer != calc_answer:
                wrong += 1

                output_layer = self.layers[-1]
                dj_dy = j_y(y, y_hat)
                dy_dz = y_z(output_layer.pre_act_last_output)
                dj_dz = dy_dz * dj_dy.T
                changes = dj_dz * output_layer.last_inputs.T * output_layer.learning_rate

                output_layer_weights_wout_bias = output_layer.weights[:, 1:]
                middle_layer = self.layers[-2]
                middle_d_relu = np.matrix([0 if o < 0 else 1 for o in vec_to_list(middle_layer.pre_act_last_output)])
                filtered_output_weights = np.multiply(middle_d_relu, output_layer_weights_wout_bias)
                b = filtered_output_weights.T * dj_dz
                middle_changes = (b * middle_layer.last_inputs.T) * middle_layer.learning_rate

                #if len(self.layers) > 2:
                #    middle_layer_weights_wout_bias = middle_layer.weights[:, 1:]
                #    back_layer = self.layers[-3]
                #    back_d_relu = np.matrix([0 if o < 0 else 1 for o in vec_to_list(back_layer.pre_act_last_output)])
                #    filtered_middle_weights = np.multiply(back_d_relu, middle_layer_weights_wout_bias)
                #    dj_du = dj_dz.T * filtered_output_weights
                #    dj_dv = dj_du * filtered_middle_weights
                #    back_gradients = (back_layer.last_inputs * dj_dv).T
                #    back_changes = back_gradients * back_layer.learning_rate
                #    adjusted_back_changes = beta_changes(back_changes, old_back_changes, self.momentum)
                #    back_layer.weights = back_layer.weights - adjusted_back_changes
                #    old_back_changes = adjusted_back_changes

                adjusted_output_changes = beta_changes(changes, old_output_changes, self.momentum)
                output_layer.weights = output_layer.weights - adjusted_output_changes
                old_output_changes = adjusted_output_changes

                adjusted_middle_changes = beta_changes(middle_changes, old_middle_changes, self.momentum)
                middle_layer.weights = middle_layer.weights - adjusted_middle_changes
                old_middle_changes = adjusted_middle_changes

        return mean(mses)

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


def beta_changes(w, old_w, momentum):
    if momentum == 0 or old_w is None:
        return w
    else:
        return w * (1 - momentum) + old_w * (momentum)


def j_y(y, y_hat):
    return (-1 * y) / y_hat


def y_z_l_m(z, l, m, sum_e_z):
    e_z_l = cached_exp(z[0, l])
    e_z_m = cached_exp(z[0, m])
    if l == m:
        return (sum_e_z * e_z_l - e_z_l ** 2) / sum_e_z ** 2
    else:
        return (- e_z_l * e_z_m) / sum_e_z ** 2


def y_z(z):
    sum_e_z = sum([cached_exp(z_j) for z_j in vec_to_list(z)])
    return np.matrix([[y_z_l_m(z, l, m, sum_e_z) for l in range(z.shape[1])] for m in range(z.shape[1])])


if __name__ == '__main__':
    y = np.matrix([1, 0])
    y_hat = np.matrix([.9, .1])

    print(j_y(y, y_hat))

    z = [1, 2, 3]
    outputs = np.matrix(softmax_func(z))
    d_y_d_z = y_z(outputs)

    print(outputs)
    print(d_y_d_z)
