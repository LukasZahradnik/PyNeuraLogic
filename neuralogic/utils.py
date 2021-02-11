from .builder import Sample, Neuron
from typing import List


def already_seen(seen: List[bool], neurons: List[int]) -> bool:
    for neuron in neurons:
        if not seen[neuron]:
            return False
    return True


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
