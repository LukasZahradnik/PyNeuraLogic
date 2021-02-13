from .builder import Weight, Sample
from typing import List
import torch
from torch import Tensor
from torch.nn import Parameter, ParameterList
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros


def already_seen(seen, neurons):
    for neuron in neurons:
        if not seen[neuron]:
            return False
    return True


def to_layers(sample: Sample):
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


class NeuraLogicHelperLayer(MessagePassing):
    def __init__(self, activation, weights):
        super(NeuraLogicHelperLayer, self).__init__(aggr=NeuraLogicLayer.aggregations.get(activation, "add"))

        self.activation = NeuraLogicLayer.activations.get(activation, None)
        self.weights = weights

    def forward(self, layer_input):
        x, neurons = layer_input

        from_neuron = []
        to_neuron = []
        edge_weight = []

        for neuron in neurons:
            if neuron.weights is None:
                edge_weight = None
                for input in neuron.inputs:
                    from_neuron.append(input)
                    to_neuron.append(neuron.index)
            else:
                for input, weight in zip(neuron.inputs, neuron.weights):
                    from_neuron.append(input)
                    to_neuron.append(neuron.index)
                    edge_weight.append(self.weights[weight + 1])

        if len(from_neuron) == 0:
            return x

        edge_index = torch.tensor([from_neuron, to_neuron])
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.activation:
            out = self.activation(out)

        return x + out

    def message(self, x_j: Tensor, edge_weight=None) -> Tensor:
        out = x_j

        if edge_weight is None:
            return out

        output = []

        for x, w in zip(out, edge_weight):
            value = x * w
            output.append(value)

        return torch.stack(output)


class NeuraLogicLayer(torch.nn.Module):
    activations = {
        "Sigmoid": torch.sigmoid,
        "ReLu": torch.relu,
        "Tanh": torch.tanh,
    }

    aggregations = {
        "Average": "mean",
        "Maximum": "max",
    }

    def __init__(self, weights: List[Weight]):
        super(NeuraLogicLayer, self).__init__()

        params = [
            Parameter(torch.tensor(weight.dimensions, dtype=torch.float), requires_grad=not weight.fixed)
            for weight in weights
        ]

        self.built_layers = False
        self.neura_layers: List[NeuraLogicHelperLayer] = []
        self.weights = ParameterList(params)

        self.reset_params()

    def reset_params(self):
        self.weights[0][0] = 1

        for weight in self.weights[1:]:
            zeros(weight)

    def forward(self, sample: Sample):
        layers = to_layers(sample)

        if not self.built_layers:
            self.built_layers = True
            self.neura_layers = [NeuraLogicHelperLayer(layer[0].activation, self.weights) for layer in layers]

        x = torch.zeros((len(sample.neurons), 1))
        x[layers[0][0].index] = 1

        for n_layer, layer in zip(self.neura_layers, layers):
            x = n_layer((x, layer))
        return x[layers[-1][0].index]
