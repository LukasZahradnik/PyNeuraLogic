from typing import List
from .builder import Sample, Weight, Neuron, Model, OFFSET
import dynet as dy


class DyNetDeserializer:
    activations = {
        "Sigmoid": dy.logistic,
        "Average": dy.average,
        "Maximum": dy.emax,
        "ReLu": dy.rectify,
        "Tanh": dy.tanh,
    }

    def __init__(self, model: Model):
        self.model = dy.ParameterCollection()
        self.weights = self.deserialize_weights(model.weights)

    def deserialize_weights(self, weights: List[Weight]):
        return [weight.value if weight.fixed else self.model.add_parameters(weight.dimensions) for weight in weights]

    def build_neuron_expr(self, neuron: Neuron, neuron_expressions: List):
        return DyNetDeserializer.to_dynet_expression(neuron, neuron_expressions, self.weights)

    def build_sample(self, sample: Sample):
        dynet_neurons = [None] * len(sample.neurons)

        for neuron in sample.neurons:
            dynet_neurons[neuron.index] = self.build_neuron_expr(neuron, dynet_neurons)
        return dynet_neurons[sample.neurons[-1].index]

    @staticmethod
    def to_dynet_value(neuron: Neuron):
        dim = 1

        if hasattr(neuron.value, "__len__"):
            dim = len(neuron.value)
        if dim == 1:
            return dy.scalarInput(float(neuron.value))
        if dim > 1:
            res = dy.vecInput(dim)
            res.set(neuron.value)
            return res
        return dy.inputTensor(neuron.value, dim)

    @staticmethod
    def to_dynet_expression(neuron: Neuron, neurons: List, weights: List):
        out = None

        if neuron.inputs:
            if neuron.weights:
                for w, i in zip(neuron.weights, neuron.inputs):
                    if not out:
                        out = weights[w + OFFSET] * neurons[i]
                    else:
                        out += weights[w + OFFSET] * neurons[i]
            else:
                for i in neuron.inputs:
                    if not out:
                        out = neurons[i]
                    else:
                        out += neurons[i]
        else:
            out = DyNetDeserializer.to_dynet_value(neuron)

        if neuron.offset:
            out += weights[neuron.offset + OFFSET]

        if neuron.activation:
            if neuron.pooling:
                out = list(out)
            return DyNetDeserializer.activations[neuron.activation](out)
        return out
