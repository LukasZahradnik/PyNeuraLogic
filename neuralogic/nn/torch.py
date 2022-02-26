from typing import List, Optional, Dict, Union, Tuple
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
    activations = {
        "Sigmoid": torch.sigmoid,
        "ReLu": torch.relu,
        "Tanh": torch.tanh,
        "Average": torch.mean,
        "Maximum": torch.max,
        "Sum": torch.sum,
        "Minimum": torch.min,
    }

    initializers = {
        Initializer.NORMAL: lambda tensor, _: torch.nn.init.normal_(tensor),
        Initializer.UNIFORM: lambda tensor, settings: torch.nn.init.uniform_(
            tensor, -settings.initializer_uniform_scale / 2, settings.initializer_uniform_scale / 2
        ),
        Initializer.CONSTANT: lambda tensor, settings: torch.nn.init.constant_(tensor, settings.initializer_const),
        Initializer.LONGTAIL: longtail,
        Initializer.GLOROT: lambda tensor, _: torch.nn.init.xavier_uniform_(tensor),
        Initializer.HE: lambda tensor, _: torch.nn.init.kaiming_uniform_(tensor),
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
        self.weights = []

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
        self.hooks_set = len(self.hooks) != 0

        torch_neurons: List[Optional[torch.Tensor]] = [None] * len(sample.neurons)

        for neuron in sample.neurons:
            torch_neurons[neuron.index] = self.to_torch_expression(neuron, torch_neurons, self.model)
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

    @staticmethod
    def to_tensor_value(value) -> torch.Tensor:
        if isinstance(value, (float, int)):
            return torch.tensor([value], requires_grad=False, dtype=torch.float64)
        return torch.tensor(value, requires_grad=False, dtype=torch.float64)

    def to_torch_expression(
        self, neuron: Neuron, neurons: List[torch.Tensor], weights: torch.nn.ParameterList
    ) -> torch.Tensor:
        if neuron.inputs:
            out_scalar, out_vector, out_matrix = self.process_neuron_inputs(neuron, neurons, weights)

            if neuron.activation:
                if not neuron.pooling:
                    out = sum(out_scalar) + sum(out_vector) + sum(out_matrix)
                else:
                    if out_matrix:
                        shape = out_matrix[0].shape
                        dummy_tensor = torch.zeros((shape), requires_grad=False)

                        out = torch.stack(
                            [
                                *out_matrix,
                                *(vector + dummy_tensor for vector in out_vector),
                                *(scalar + dummy_tensor for scalar in out_scalar),
                            ]
                        )
                    elif out_vector:
                        shape = out_vector[0].shape
                        dummy_tensor = torch.zeros((shape), requires_grad=False)
                        out = torch.stack([*out_vector, *(scalar + dummy_tensor for scalar in out_scalar)])
                    else:
                        out = torch.stack(out_scalar)
                if neuron.activation != "Identity":
                    if neuron.pooling:
                        out = NeuraLogic.activations[neuron.activation](out, dim=0)
                    else:
                        out = NeuraLogic.activations[neuron.activation](out)
            else:
                out = sum(out_scalar) + sum(out_vector) + sum(out_matrix)
        else:
            out = NeuraLogic.to_tensor_value(neuron.value)

            if neuron.activation and neuron.activation != "Identity":
                if neuron.pooling:
                    out = NeuraLogic.activations[neuron.activation](out, dim=0)
                else:
                    out = NeuraLogic.activations[neuron.activation](out)

        if self.hooks_set and neuron.hook_name is not None and neuron.hook_name in self.hooks:
            self.run_hook(neuron.hook_name, out.value())

        return out

    def process_neuron_inputs(
        self, neuron: Neuron, neurons: List[torch.Tensor], weights: torch.nn.ParameterList
    ) -> Tuple[List[Union[torch.Tensor, int, float]], List[torch.Tensor], List[torch.Tensor]]:
        out_matrix = []
        out_vector = []
        out_scalar = []

        if neuron.weights:
            for w, i in zip(neuron.weights, neuron.inputs):
                neuron = neurons[i]
                weight = weights[w]

                neuron_size = neuron.size()
                weight_size = weight.size()

                if not neuron_size or not weight_size or neuron_size == (1,) or weight_size == (1,):
                    result = torch.multiply(weight, neuron)
                else:
                    result = torch.matmul(weight, neuron)

                res_size = result.size()
                if not res_size or res_size == (1,):
                    out_scalar.append(result)
                elif len(res_size) == 1:
                    out_vector.append(result)
                else:
                    out_matrix.append(result)
        else:
            for i in neuron.inputs:
                result = neurons[i]

                if isinstance(result, (int, float)):
                    out_scalar.append(result)
                else:
                    res_size = result.size()
                    if not res_size or res_size == (1,):
                        out_scalar.append(result)
                    elif len(res_size) == 1:
                        out_vector.append(result)
                    else:
                        out_matrix.append(result)
        return out_scalar, out_vector, out_matrix
