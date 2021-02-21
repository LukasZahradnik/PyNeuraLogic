from .builder import Sample, Neuron
from typing import List


def already_seen(seen: List[bool], neurons: List[int]) -> bool:
    for neuron in neurons:
        if not seen[neuron]:
            return False
    return True


def to_layers(sample: Sample) -> List[List[Neuron]]:
    seen = [False] * len(sample.neurons)
    current_seen = []

    layers = []
    layer = []

    for neuron in sample.neurons:
        if already_seen(seen, neuron.inputs):
            layer.append(neuron)
            current_seen.append(neuron.index)
        else:
            layers.append(layer)

            for n in current_seen:
                seen[n] = True

            layer = [neuron]
            current_seen = [neuron.index]
    else:
        layers.append(layer)
    return layers
