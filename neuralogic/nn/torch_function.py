import json
from typing import Callable, Any, List

import torch
from torch import nn
from torch.autograd import Function

from neuralogic.core import Template, Settings
from neuralogic.core.constructs.java_objects import ValueFactory
from neuralogic.dataset import Dataset


class _NeuraLogicFunction(Function):
    @staticmethod
    def forward(ctx, mapping, value_factory, sample, model, number_format, dtype, *inputs):
        ctx.model = model
        ctx.sample = sample
        ctx.number_format = number_format
        ctx.dtype = dtype
        ctx.mapping = mapping

        for fact, value in mapping.items():
            sample.set_fact_value(fact, value_factory.get_value(value)[1])

        return torch.tensor(model(sample, train=False), dtype=dtype, requires_grad=True)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        model = ctx.model
        sample = ctx.sample
        number_format = ctx.number_format
        dtype = ctx.dtype

        backproper, weight_updater = model.backprop(sample, -grad_outputs[0].detach().numpy())
        state_index = backproper.stateIndex

        gradients = tuple(
            -torch.tensor(
                json.loads(
                    str(sample.get_fact(fact).getComputationView(state_index).getGradient().toString(number_format))
                ),
                dtype=dtype,
                requires_grad=False,
            ).reshape(input.shape)
            for fact, input in ctx.mapping.items()
        )

        trainer = model.strategy.getTrainer()
        trainer.updateWeights(model.strategy.getCurrentModel(), weight_updater)
        trainer.invalidateSample(trainer.getInvalidation(), sample.java_sample)

        return None, None, None, None, None, None, *gradients


class NeuraLogic(nn.Module):
    def __init__(
        self,
        template: Template,
        input_facts: List,
        output_relation,
        to_logic: Callable,
        settings: Settings,
        dtype=torch.float32,
    ):
        super(NeuraLogic, self).__init__()

        settings.iso_value_compression = False
        settings.chain_pruning = False

        self.dtype = dtype
        self.to_logic = to_logic

        self.model = template.build(settings)
        self.number_format = self.model.settings.settings_class.superDetailedNumberFormat

        dataset = Dataset([input_facts], [output_relation])
        self.sample = self.model.build_dataset(dataset, learnable_facts=True).samples[0]
        self.value_factory = ValueFactory()

        self.internal_weights = nn.Parameter(torch.empty((0,)))
        self.neuralogic_function = _NeuraLogicFunction

    def forward(self, *inputs, **kwargs):
        mapping = self.to_logic(*inputs, **kwargs)

        return _NeuraLogicFunction.apply(
            mapping,
            self.value_factory,
            self.sample,
            self.model,
            self.number_format,
            self.dtype,
            *(value for value in mapping.values()),
        )
