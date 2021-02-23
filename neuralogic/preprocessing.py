from typing import List
from .utils import to_layers
from .builder import Sample, Neuron
from neuralogic.error import MixedActivationFunctionsInLayerException


def unit_weight_generator():
    while True:
        yield -1


class Layer:
    def __init__(self, layer: List[Neuron]):
        self.u = []
        self.v = []
        self.weights = []
        self.targets = []

        self.has_weights = False
        self.activation = -1

        for neuron in layer:
            if self.activation != -1 and neuron.activation != self.activation:
                raise MixedActivationFunctionsInLayerException()

            self.activation = neuron.activation
            self.targets.append(neuron.index)

            weights = unit_weight_generator()
            if neuron.weights is not None and len(neuron.weights) != 0:
                weights = neuron.weights
                self.has_weights = True

            for input, weight in zip(neuron.inputs, weights):
                self.u.append(input)
                self.v.append(neuron.index)
                self.weights.append(weight)


class ProcessedSample:
    def __init__(self, sample: Sample):
        self.sample = sample
        self.layers = [Layer(layer) for layer in to_layers(sample)]

    @property
    def target(self):
        return float(self.sample.target)

    @property
    def output_neuron(self):
        return self.sample.output_neuron

    @property
    def neurons(self):
        return self.sample.neurons


def to_processed_samples(samples: List[Sample]) -> List[ProcessedSample]:
    return [ProcessedSample(sample) for sample in samples]
