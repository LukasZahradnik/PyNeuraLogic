from collections.abc import Collection
from typing import Any

import torch

from neuralogic.core.constructs.java_objects import ValueFactory
from neuralogic.core.torch.network_output import PyNeuraLogicNetworkOutput
from neuralogic.core.torch.tensor import NeuralogicOptTensor


class TorchNeuralModule:
    """
    Mixin class providing utility methods for integrating PyNeuraLogic with PyTorch.
    It handles the conversion of Java weights to PyTorch tensors and the management of learnable parameters.
    """

    def tensor_parameters(
        self,
        tensor_parameters: list[NeuralogicOptTensor] | None,
        weight_updater: Any,
        value_factory: Any,
        neural_model: Any,
    ) -> list[NeuralogicOptTensor]:
        """
        Creates or retrieves PyTorch-compatible tensor parameters from the Java neural model.

        Parameters
        ----------
        tensor_parameters : List[NeuralogicOptTensor], optional
            Already existing tensor parameters. If None, they will be created.
        weight_updater : Any
            Weight updater object.
        value_factory : Any
            Value factory for Java to Python conversion.
        neural_model : Any
            Java neural model containing the weights.

        Returns
        -------
        List[NeuralogicOptTensor]
            List of learnable tensor parameters.
        """
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

    def update_tensor_parameters(self, tensor_parameters: list[NeuralogicOptTensor] | None) -> None:
        """
        Updates the data of the provided tensor parameters from their underlying Java weights.

        Parameters
        ----------
        tensor_parameters : List[NeuralogicOptTensor]
            The list of tensor parameters to update.
        """
        if tensor_parameters is None:
            return

        for param in tensor_parameters:
            param.data = torch.tensor(ValueFactory.from_java(param._neuralogic_weight.value))

    def forward(self, model: Any, samples: Any, results: Any) -> PyNeuraLogicNetworkOutput:
        """
        Performs the forward pass by wrapping the results in PyNeuraLogicNetworkOutput.

        Parameters
        ----------
        model : Any
            The neural model.
        samples : Any
            The samples used for the forward pass.
        results : Any
            The raw results from the Java engine.

        Returns
        -------
        PyNeuraLogicNetworkOutput
            The wrapped output compatible with PyTorch autograd.
        """
        if not isinstance(samples, Collection):
            return PyNeuraLogicNetworkOutput.apply(
                samples, model, torch.tensor(results[0], dtype=torch.float, requires_grad=True)
            )
        return PyNeuraLogicNetworkOutput.apply(
            samples, model, torch.tensor(results, dtype=torch.float, requires_grad=True)
        )
