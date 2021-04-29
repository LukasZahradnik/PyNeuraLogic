from typing import List, Optional
from neuralogic.core.builder import Sample, Weight, Neuron
import dynet as dy

from neuralogic.core.settings import Settings


class NeuraLogicLayer:
    activations = {
        "Sigmoid": dy.logistic,
        "Average": dy.average,
        "Maximum": dy.emax,
        "ReLu": dy.rectify,
        "Tanh": dy.tanh,
    }

    def __init__(self, model, settings: Optional[Settings] = None):
        if settings is None:
            settings = Settings()

        self.model = dy.ParameterCollection()
        self.weights = self.deserialize_weights(model)
        self.settings = settings

    def deserialize_weights(self, weights: List[Weight]) -> List[dy.Parameters]:
        return [
            weight.value if weight.fixed else self.model.add_parameters(weight.dimensions, init="uniform")
            for weight in weights
        ]

    def build_sample(self, sample: Sample) -> dy.Expression:
        dynet_neurons: List[Optional[dy.Expression]] = [None] * len(sample.neurons)

        for neuron in sample.neurons:
            dynet_neurons[neuron.index] = NeuraLogicLayer.to_dynet_expression(neuron, dynet_neurons, self.weights)
        return dynet_neurons[sample.neurons[-1].index]

    def __call__(self, sample: Sample) -> dy.Expression:
        return self.build_sample(sample)

    @staticmethod
    def to_dynet_value(value) -> dy.Expression:
        dim = 1
        if hasattr(value, "__len__"):
            dim = len(value)
        if dim == 1:
            return dy.scalarInput(float(value))
        if dim > 1:
            res = dy.vecInput(dim)
            res.set(value)
            return res
        return dy.inputTensor(value, dim)

    @staticmethod
    def to_dynet_expression(neuron: Neuron, neurons: List[dy.Expression], weights: List):
        if neuron.inputs:
            out = NeuraLogicLayer.process_neuron_inputs(neuron, neurons, weights)
        else:
            out = NeuraLogicLayer.to_dynet_value(neuron.value)

        if neuron.activation:
            if neuron.pooling:
                out = list(out)
            out = NeuraLogicLayer.activations[neuron.activation](out)
        return out

    @staticmethod
    def process_neuron_inputs(neuron: Neuron, neurons: List, weights: List[dy.Parameters]) -> dy.Expression:
        if neuron.weights:
            if neurons[neuron.inputs[0]].dim()[0] == (1,) and not isinstance(weights[neuron.weights[0]], int):
                out = dy.cmult(neurons[neuron.inputs[0]], weights[neuron.weights[0]])
            else:
                out = weights[neuron.weights[0]] * neurons[neuron.inputs[0]]
            for w, i in zip(neuron.weights[1:], neuron.inputs[1:]):
                if neurons[i].dim()[0] == (1,) and not isinstance(weights[w], int):
                    out += dy.cmult(weights[w], neurons[i])
                else:
                    out += weights[w] * neurons[i]
        else:
            out = neurons[neuron.inputs[0]]
            for i in neuron.inputs[1:]:
                out += neurons[i]
        return out
