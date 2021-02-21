import dgl
import dgl.function as fn
import torch
from typing import List
from torch.nn import Parameter, ParameterList, Sequential
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

    def __init__(self, layer: Layer, neuron_count: int, weights: ParameterList):
        super(NeuraLogicHelperLayer, self).__init__()

        self.reduce = "Sum" if layer.activation not in NeuraLogicHelperLayer.activations else layer.activation

        self.neuron_count = neuron_count
        self.layer = layer
        self.weights = weights

    def forward(self, layer_input: torch.Tensor) -> torch.Tensor:
        x = layer_input

        if len(self.layer.u) == 0:
            return x

        old_x = x.clone()
        old_x[self.layer.targets] = 0

        g = dgl.graph((self.layer.u, self.layer.v), num_nodes=self.neuron_count)

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
        return {"m": torch.stack([x * self.weights[w] for x, w in zip(xs, self.layer.weights)])}


class NeuraLogicLayer(torch.nn.Module):
    def __init__(self, weights: List[Weight]):
        super(NeuraLogicLayer, self).__init__()

        params = [
            Parameter(torch.tensor(weight.value, dtype=torch.float), requires_grad=not weight.fixed)
            for weight in weights
        ]

        self.weights = ParameterList(params)
        self.reset_params(weights)

    def reset_params(self, meta_weights: List[Weight]):
        for weight, meta_weight in zip(self.weights, meta_weights):
            if not meta_weight.fixed:
                torch.nn.init.uniform_(weight)

    def forward(self, sample: ProcessedSample):
        neuron_count = len(sample.neurons)

        neural_layers = [NeuraLogicHelperLayer(layer, neuron_count, self.weights) for layer in sample.layers]
        model = Sequential(*neural_layers)

        x = torch.zeros((neuron_count, 1))

        for neuron in sample.neurons:
            if neuron.value is not None:
                x[neuron.index] = neuron.value

        x = model(x)
        return torch.take(x, torch.tensor([sample.neurons[-1].index]))
