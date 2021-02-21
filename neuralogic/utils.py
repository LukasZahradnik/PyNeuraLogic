from .builder import Sample, Neuron
from typing import List


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
