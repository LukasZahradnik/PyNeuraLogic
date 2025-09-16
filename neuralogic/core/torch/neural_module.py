from typing import List, Collection

import torch

from neuralogic.core.constructs.java_objects import ValueFactory
from neuralogic.core.torch.network_output import PyNeuraLogicNetworkOutput
from neuralogic.core.torch.tensor import NeuralogicOptTensor


class TorchNeuralModule:
    def tensor_parameters(
        self, tensor_parameters, weight_updater, value_factory, neural_model
    ) -> List[NeuralogicOptTensor]:
        if tensor_parameters is None:
            tensor_parameters = [
                NeuralogicOptTensor.create(
                    weight,
                    ValueFactory.from_java(weight.value),
                    weight_updater,
                    value_factory,
                )
                for weight in neural_model.getAllWeights()
                if weight.isLearnable
            ]

        return tensor_parameters

    def update_tensor_parameters(self, tensor_parameters):
        if tensor_parameters is None:
            return

        for param in tensor_parameters:
            param.data = torch.tensor(ValueFactory.from_java(param._neuralogic_weight.value))

    def forward(self, model, samples, results) -> PyNeuraLogicNetworkOutput:
        if not isinstance(samples, Collection):
            return PyNeuraLogicNetworkOutput.apply(
                samples, model, torch.tensor(results[0], dtype=torch.float, requires_grad=True)
            )
        return PyNeuraLogicNetworkOutput.apply(
            samples, model, torch.tensor(results, dtype=torch.float, requires_grad=True)
        )
