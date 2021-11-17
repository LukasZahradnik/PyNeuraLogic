from typing import List, Optional, Dict, Union
import torch
import numpy as np

from neuralogic.core.builder import Sample, Weight, Neuron
from neuralogic.core.settings import SettingsProxy
from neuralogic.core.enums import Initializer, Backend
from neuralogic.nn.base import AbstractNeuraLogic


def longtail(tensor: torch.Tensor, _: SettingsProxy):
    init = np.random.rand(*tensor.size())

    x0, x1, power = 0.0, 10.0, 50.0
    x0_power = np.power(x0, power + 1)

    init = x1 - np.power((np.power(x1, power + 1) - x0_power) * init + x0_power, 1 / (power + 1))
    tensor.fill_(init)


class NeuraLogic(AbstractNeuraLogic):

    initializers = {
        Initializer.NORMAL: lambda tensor, _: torch.nn.init.normal_(tensor),
        Initializer.UNIFORM: lambda tensor, settings: torch.nn.init.uniform_(
            tensor, -settings.initializer_uniform_scale / 2, settings.initializer_uniform_scale / 2
        ),
        Initializer.CONSTANT: lambda tensor, settings: torch.nn.init.constant_(tensor, settings.initializer_const),
        Initializer.LONGTAIL: longtail,
        # TODO: GLOROT and HE
    }

    def __init__(self, model: List[Weight], template, settings: Optional[SettingsProxy] = None):
        super().__init__(Backend.TORCH, template, settings)

        self.model: Optional[torch.nn.ParameterList] = None
        self.weights_meta = model
        self.weights: List[torch.nn.Parameter] = []

        self.reset_parameters()

    def reset_parameters(self):
        initializer = Initializer[str(self.settings.initializer)]

        if initializer not in NeuraLogic.initializers:
            raise NotImplementedError

        weight_initializer = NeuraLogic.initializers[initializer]

        for weight in self.weights_meta:
            if weight.fixed:
                parameter = torch.nn.Parameter(torch.tensor(weight.value, dtype=torch.float64), requires_grad=False)
            else:
                parameter = torch.nn.Parameter(
                    torch.empty(size=weight.dimensions, requires_grad=True, dtype=torch.float64)
                )
                weight_initializer(parameter.data, self.settings)
            self.weights.append(parameter)
        self.model = torch.nn.ParameterList(self.weights)

    def build_sample(self, sample: Sample):
        torch_neurons: List[torch.Tensor] = [torch.ones((1, 1))] * (len(sample.neurons) + 1)

        for neuron in sample.neurons:
            torch_neurons[neuron.index] = to_torch_expression(
                neuron.inputs,
                neuron.weights if neuron.weights is not None else [],
                neuron.activation,
                neuron.pooling,
                neuron.value,
                torch_neurons,
                self.model,
            )
        return torch_neurons[sample.neurons[-1].index]

    def __call__(self, sample: Sample):
        return self.build_sample(sample)

    def state_dict(self) -> Dict:
        weights = {}

        for meta, weight in zip(self.weights_meta, self.weights):
            if not meta.fixed and meta.index >= 0:
                weights[meta.index] = weight if isinstance(weight, (int, float)) else weight.data
        return {"weights": weights}

    def load_state_dict(self, state_dict: Dict):
        weight_dict = state_dict["weights"]

        for i, (meta, weight) in enumerate(zip(self.weights_meta, self.weights)):
            if meta.index < 0 or meta.fixed:
                continue

            if isinstance(weight, (int, float)):
                weight.data = torch.tensor([weight_dict[meta.index]], dtype=torch.float64)
            else:
                weight.data = torch.tensor(weight_dict[meta.index], dtype=torch.float64)


@torch.jit.script
def to_tensor_value(value: Optional[Union[List[int], float, int]]) -> torch.Tensor:
    if value is None:
        return torch.tensor([1.0], requires_grad=False, dtype=torch.float64)
    if isinstance(value, float):
        return torch.tensor([value], requires_grad=False, dtype=torch.float64)
    if isinstance(value, int):
        return torch.tensor([value], requires_grad=False, dtype=torch.float64)
    tensor_val: List[int] = value

    return torch.tensor(tensor_val, requires_grad=False, dtype=torch.float64)


@torch.jit.script
def process_neuron_inputs(
    neuron_weights: List[int], neuron_inputs: List[int], neurons: List[torch.Tensor], weights: List[torch.Tensor]
) -> torch.Tensor:
    out = []

    if neuron_weights:
        for w, i in zip(neuron_weights, neuron_inputs):
            neuron = neurons[i]
            weight = weights[w]

            neuron_size = neuron.size()
            weight_size = weight.size()

            if not neuron_size or not weight_size or neuron_size == (1, 1) or weight_size == (1, 1):
                out.append(torch.multiply(weight, neuron))
            else:
                out.append(torch.matmul(weight, neuron))
    else:
        out = [neurons[i] for i in neuron_inputs]

    return torch.stack(out)


@torch.jit.script
def evaluate_activation(value: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "Sigmoid":
        return torch.sigmoid(value)
    if activation == "Average":
        return torch.mean(value)
    if activation == "Maximum":
        return torch.max(value)
    if activation == "ReLu":
        return torch.relu(value)
    if activation == "Tanh":
        return torch.tanh(value)
    return value


@torch.jit.script
def to_torch_expression(
    neuron_inputs: List[int],
    neuron_weights: List[int],
    activation: Optional[str],
    pooling: bool,
    value: Optional[Union[List[int], float, int]],
    neurons: List[torch.Tensor],
    weights: List[torch.Tensor],
) -> torch.Tensor:
    if neuron_inputs:
        out: torch.Tensor = process_neuron_inputs(neuron_weights, neuron_inputs, neurons, weights)

        if activation is not None:
            if not pooling:
                tensor: torch.Tensor = out
                out = torch.sum(tensor)
            if activation != "Identity":
                out = evaluate_activation(out, activation)
            return out
        else:
            return torch.sum(torch.tensor(out))
    else:
        new_out = to_tensor_value(value)

        if activation is not None and activation != "Identity":
            return evaluate_activation(new_out, activation)
        return new_out
