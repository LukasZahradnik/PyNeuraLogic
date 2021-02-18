import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch.nn import Parameter, ParameterList
from neuralogic.builder import Neuron, Sample, Weight
from neuralogic.utils import to_layers


class NeuraLogicHelperLayer(nn.Module):
    reducers = {
        "Average": fn.mean(msg="m", out="x"),
        "Maximum": fn.max(msg="m", out="x"),
        "Sum": fn.sum(msg="m", out="x"),
    }

    activations = {
        "Sigmoid": torch.sigmoid,
        "ReLu": torch.relu,
        "Tanh": torch.tanh,
    }

    def __init__(self, activation, reduce, weights):
        super(NeuraLogicHelperLayer, self).__init__()

        self.activation = NeuraLogicHelperLayer.activations.get(activation, None)
        self.reduce = NeuraLogicHelperLayer.reducers.get(reduce, NeuraLogicHelperLayer.reducers["Sum"])

        self.weights = weights
        self.temp_weights = None

    def forward(self, layer_input):
        x, neurons, num_nerons = layer_input

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

        g = dgl.graph((from_neuron, to_neuron), num_nodes=num_nerons)
        self.temp_weights = edge_weight

        with g.local_scope():
            g.ndata["x"] = x
            g.update_all(self.message, self.reduce)
            return g.ndata["x"]

    def message(self, edges):
        xs = edges.src["x"]

        if self.temp_weights is None or len(self.temp_weights) == 0:
            if self.activation is None:
                return {"m": xs}
            return {"m": self.activation(xs)}

        output = []

        for x, w in zip(xs, self.temp_weights):
            value = x * w
            output.append(value)

        if self.activation is None:
            return {"m": torch.stack(output)}
        return {"m": self.activation(torch.stack(output))}


class NeuraLogicLayer(torch.nn.Module):
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
            weight.data.fill_(0)

    def forward(self, sample: Sample):
        layers = to_layers(sample)

        if not self.built_layers:
            self.built_layers = True
            self.neura_layers = []

            for i, layer in enumerate(layers):
                if i != 0:
                    self.neura_layers.append(
                        NeuraLogicHelperLayer(layers[i - 1][0].activation, layer[0].activation, self.weights)
                    )
                else:
                    self.neura_layers.append(NeuraLogicHelperLayer(None, layer[0].activation, self.weights))

        x = torch.zeros((len(sample.neurons), 1))
        x[layers[0][0].index] = 1

        for n_layer, layer in zip(self.neura_layers, layers):
            x = x + n_layer((x, layer, len(sample.neurons)))
        return x[layers[-1][0].index]
