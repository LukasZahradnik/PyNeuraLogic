from .builder import Sample, Neuron
from typing import List
from neuralogic.error import MixedActivationFunctionsInLayerException, MixedWeightsAndNoWeightsInLayerException


def already_seen(seen: List[bool], neurons: List[int]) -> bool:
    for neuron in neurons:
        if not seen[neuron]:
            return False
    return True


def validate_layer(layer: List[Neuron]):
    if len(layer) == 0:
        return

    activation = layer[0].activation
    weights = layer[0].weights

    for neuron in layer[1:]:
        if neuron.activation != activation:
            raise MixedActivationFunctionsInLayerException()
        if weights == neuron.weights or (weights is not None and neuron.weights is not None):
            continue
        raise MixedWeightsAndNoWeightsInLayerException()


def to_layers(sample: Sample) -> List[List[Neuron]]:
    seen = [False] * len(sample.neurons)
    layers = []

    while True:
        index = 0
        current_layer = []

        for i in range(index, len(sample.neurons)):
            neuron = sample.neurons[i]
            if seen[neuron.index]:
                continue
            if not already_seen(seen, neuron.inputs):
                if index == 0:
                    index = i
                continue
            current_layer.append(neuron)
        layers.append(current_layer)
        for n in current_layer:
            seen[n.index] = True
        if index == 0:
            return layers
