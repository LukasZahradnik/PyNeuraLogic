import json
from typing import Callable

import torch
from torch import nn
from torch.autograd import Function

from neuralogic.core import Template, Settings
from neuralogic.dataset import Dataset


class _NeuraLogicFunction(Function):
    @staticmethod
    def forward(ctx, model, mapping, number_format, dtype, *inputs):
        ctx.mapping = mapping
        ctx.model = model
        ctx.number_format = number_format
        ctx.inputs = inputs
        ctx.dtype = dtype

        example_map, query_map = mapping

        dataset = Dataset(
            [[example[value.detach().numpy()] for example, value in example_map.items()]],
            [query_map],
        )

        built_dataset = model.build_dataset(dataset)
        sample = built_dataset.samples[0]

        ctx.sample = sample

        return torch.tensor(model(sample, train=False), dtype=dtype, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        backproper, weight_updater = ctx.model.backprop(ctx.sample, grad_output.detach().numpy())
        state_index = backproper.stateIndex

        mapping = ctx.mapping
        sample = ctx.sample
        number_format = ctx.number_format
        dtype = ctx.dtype

        gradients = tuple(
            torch.tensor(
                json.loads(str(sample.get_fact(fact).getComputationView(-1).getGradient().toString(number_format))),
                dtype=dtype,
            ).reshape(input.shape)
            for fact, input in zip(mapping[0], ctx.inputs)
        )

        trainer = ctx.model.strategy.getTrainer()

        trainer.updateWeights(ctx.model.strategy.getCurrentModel(), weight_updater)
        trainer.invalidateSample(trainer.getInvalidation(), sample.java_sample)

        return None, None, None, None, *gradients


class NeuraLogic(nn.Module):
    def __init__(self, template: Template, to_logic: Callable, settings: Settings, dtype=torch.float32):
        super(NeuraLogic, self).__init__()

        # settings.iso_value_compression = False
        # settings.chain_pruning = False

        self.model = template.build(settings)
        self.number_format = self.model.settings.settings_class.superDetailedNumberFormat
        self.dtype = dtype

        self.internal_weights = nn.Parameter(torch.empty((0,)))
        self.to_logic = to_logic

    def forward(self, *inputs):
        mapping = self.to_logic(*inputs)

        return _NeuraLogicFunction.apply(
            self.model, mapping, self.number_format, self.dtype, *(value for value in mapping[0].values())
        )
