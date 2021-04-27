from typing import List, Union, Tuple
from neuralogic.core.builder import Sample, Neuron
from neuralogic.error import MixedActivationFunctionsInLayerException


def unit_weight_generator():
    while True:
        yield -1


def to_layers(sample: Sample) -> List[List[Neuron]]:
    neuron_layers = [0] * len(sample.neurons)
    layers: List[List[Neuron]] = [[]]

    for neuron in sample.neurons:
        if len(neuron.inputs) == 0:
            layers[0].append(neuron)
            continue

        layer = max(neuron_layers[input] for input in neuron.inputs) + 1
        if len(layers) <= layer:
            layers.append([neuron])
            neuron_layers[neuron.index] = len(layers)
            continue

        for layer in layers[layer:]:
            if layer[0].activation == neuron.activation:
                layer.append(neuron)
                neuron_layers[neuron.index] = neuron_layers[layer[0].index]
                break
        else:
            layers.append([neuron])
            neuron_layers[neuron.index] = len(layers)
    return layers


class Layer:
    def __init__(self, layer: List[Neuron]):
        self.u = []
        self.v = []
        self.weights = []
        self.targets = []

        self.has_weights = False
        self.activation = -1
        self.value_shape: Union[Tuple[int], Tuple[int, int]] = (1,)

        for neuron in layer:
            if self.activation != -1 and neuron.activation != self.activation:
                raise MixedActivationFunctionsInLayerException()

            self.activation = neuron.activation
            self.targets.append(neuron.index)

            if neuron.value:
                if isinstance(neuron.value, list):
                    if isinstance(neuron.value[0], list):
                        self.value_shape = len(neuron.value), len(neuron.value[0])
                    else:
                        self.value_shape = (len(neuron.value),)
                else:
                    self.value_shape = (1,)

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
