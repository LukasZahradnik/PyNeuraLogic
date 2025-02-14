from typing import Any, Collection

import torch
from torch.autograd import Function


class PyNeuraLogicNetworkOutput(Function):
    @staticmethod
    def forward(ctx, samples, model, tensor):
        ctx.model = model
        ctx.samples = samples
        ctx.shape = tensor.shape

        return tensor

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        model = ctx.model
        samples = ctx.samples

        if not isinstance(samples, Collection):
            model._backprop(samples, -grad_outputs[0])
        else:
            for sample, grad in zip(samples, grad_outputs[0]):
                model._backprop(sample, -grad)
        return None, None, torch.zeros(ctx.shape, dtype=torch.float64)
