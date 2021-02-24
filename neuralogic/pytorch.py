from neuralogic.builder import Weight
from typing import List
import torch
from torch import Tensor
from torch.nn import Parameter, ParameterList, Sequential
from torch_geometric.nn.conv import MessagePassing
from neuralogic.preprocessing import ProcessedSample, Layer


class NeuraLogicHelperLayer(MessagePassing):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.layer.u) == 0:
            return x

        old_x = x.clone()
        edge_index = torch.tensor([self.layer.u, self.layer.v])

        if self.layer.has_weights:
            print(x.shape)
            out = self.propagate(edge_index, x=x)
        else:
            out = x

        if self.reduce != "Sum":
            out = self.reduce_function(out)

        if len(old_x.shape) < len(out):
            old_x = old_x.reshape(old_x.shape + tuple(1 for _ in range(len(out.shape) - len(old_x.shape))))
        return old_x

    def reduce_function(self, x):
        return self.activations[self.reduce](x)

    def message(self, x_j: Tensor) -> Tensor:
        out = [
            (self.weights[w] @ x) if x.shape != (1,) and self.weights[w].shape != (1,) else (x * self.weights[w])
            for x, w in zip(x_j, self.layer.weights)
        ]

        return torch.stack(out)


class NeuraLogicLayer(torch.nn.Module):
    def __init__(self, weights: List[Weight]):
        super(NeuraLogicLayer, self).__init__()

        params = [
            Parameter(
                torch.tensor(weight.value, dtype=torch.float).reshape(weight.dimensions), requires_grad=not weight.fixed
            )
            for weight in weights
        ]

        self.weights = ParameterList(params)
        self.meta_weights = weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight, meta_weight in zip(self.weights, self.meta_weights):
            if not meta_weight.fixed:
                torch.nn.init.uniform_(weight)

    def forward(self, sample: ProcessedSample):
        neuron_count = len(sample.neurons)

        neural_layers = [NeuraLogicHelperLayer(layer, neuron_count, self.weights) for layer in sample.layers]
        model = Sequential(*neural_layers)

        x = torch.zeros((neuron_count, 3, 1))

        for neuron_index in sample.layers[0].targets:
            neuron = sample.neurons[neuron_index]
            x[neuron.index].fill_(neuron.value)

        x = model(x)
        return torch.take(x, torch.tensor([sample.neurons[-1].index]))
