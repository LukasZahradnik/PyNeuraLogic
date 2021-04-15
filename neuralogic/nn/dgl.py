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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.layer.u) == 0:
            return x

        # old_x = x.clone()
        # old_x[self.layer.targets] = 0

        g = dgl.graph((self.layer.u, self.layer.v), num_nodes=self.neuron_count)
        message = self.message if self.layer.has_weights else fn.copy_src(src="x", out="m")
        reduce = fn.sum(msg="m", out="x") if self.reduce == "Sum" else self.reduce_function

        with g.local_scope():
            g.ndata["x"] = x
            g.update_all(message, reduce)

            # if len(old_x.shape) < len(g.ndata["x"]):
            #     old_x = old_x.reshape(old_x.shape + tuple(1 for _ in range(len(g.ndata["x"].shape) - len(old_x.shape))))

            return g.ndata["x"]

    def reduce_function(self, nodes):
        x = torch.sum(nodes.mailbox["m"], dim=1)
        out = self.activations[self.reduce](x)

        return {"x": out}

    def message(self, edges):
        xs = edges.src["x"]
        out = [
            (self.weights[w] @ x) if x.shape != (1,) and self.weights[w].shape != (1,) else (x * self.weights[w])
            for x, w in zip(xs, self.layer.weights)
        ]

        return {"m": torch.stack(out)}


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

        x = torch.zeros((neuron_count, *sample.layers[0].value_shape))

        for neuron_index in sample.layers[0].targets:
            neuron = sample.neurons[neuron_index]
            x[neuron.index] = torch.tensor(neuron.value)

        x = model(x)
        return torch.take(x, torch.tensor([sample.neurons[-1].index]))
