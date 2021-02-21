import dgl
import dgl.function as fn
import torch
from typing import List, Tuple
from torch.nn import Parameter, ParameterList
from neuralogic.builder import Weight
from neuralogic.preprocessing import ProcessedSample, Layer


class NeuraLogicHelperLayer(torch.nn.Module):
    activations = {
        "Sigmoid": torch.sigmoid,
        "ReLu": torch.relu,
        "Tanh": torch.tanh,
        "Average": lambda x: torch.mean(x, 1, True),
        "Maximum": lambda x: torch.max(x, 1, True),
    }

    def __init__(self, layer, weights):
        super(NeuraLogicHelperLayer, self).__init__()

        self.reduce = "Sum" if layer.activation not in NeuraLogicHelperLayer.activations else layer.activation
        self.weights = weights
        self.temp_weights = None

    def forward(self, layer_input: Tuple[torch.Tensor, Layer, int]):
        x, layer, num_nerons = layer_input

        if len(layer.u) == 0:
            return x

        old_x = x.clone()
        old_x[layer.targets] = 0

        g = dgl.graph((layer.u, layer.v), num_nodes=num_nerons)
        self.temp_weights = layer.weights

        message = self.message
        reduce = fn.sum(msg="m", out="x") if self.reduce == "Sum" else self.reduce_function

        with g.local_scope():
            g.ndata["x"] = x
            g.update_all(message, reduce)
            return old_x + g.ndata["x"]

    def reduce_function(self, nodes):
        x = torch.sum(nodes.mailbox["m"], dim=1)
        out = self.activations[self.reduce](x)

        return {"x": out}

    def message(self, edges):
        xs = edges.src["x"]
        return {"m": torch.stack([x * self.weights[w] for x, w in zip(xs, self.temp_weights)])}


class NeuraLogicLayer(torch.nn.Module):
    def __init__(self, weights: List[Weight]):
        super(NeuraLogicLayer, self).__init__()

        params = [
            Parameter(torch.tensor(weight.value, dtype=torch.float), requires_grad=not weight.fixed)
            for weight in weights
        ]

        self.built_layers = False
        self.neura_layers: List[NeuraLogicHelperLayer] = []
        self.weights = ParameterList(params)

        self.reset_params(weights)

    def reset_params(self, meta_weights: List[Weight]):
        for weight, meta_weight in zip(self.weights, meta_weights):
            if not meta_weight.fixed:
                torch.nn.init.uniform_(weight)

    def forward(self, sample: ProcessedSample):
        if not self.built_layers:
            self.built_layers = True
            self.neura_layers = [NeuraLogicHelperLayer(layer, self.weights) for layer in sample.layers]

        x = torch.zeros((len(sample.neurons), 1))

        for neuron in sample.neurons:
            if neuron.value is not None:
                x[neuron.index] = neuron.value
        for n_layer, layer in zip(self.neura_layers, sample.layers):
            yy = n_layer((x, layer, len(sample.neurons)))
            x = yy

        return torch.take(x, torch.tensor([sample.neurons[-1].index]))
