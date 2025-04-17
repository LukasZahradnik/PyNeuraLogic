from typing import Callable, Any, List, Union

import torch
from torch import nn
from torch.autograd import Function

from neuralogic.core import Template, Settings
from neuralogic.core.constructs.java_objects import ValueFactory
from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.core.constructs.rule import Rule
from neuralogic.dataset import Dataset, Sample


class _NeuraLogicFunction(Function):
    @staticmethod
    def forward(ctx, mapping, value_factory, sample, model, dtype, *inputs):
        ctx.model = model
        ctx.sample = sample
        ctx.dtype = dtype
        ctx.mapping = mapping

        for fact in mapping:
            sample.set_fact_value(fact, value_factory.get_value(fact.weight)[1])

        return torch.tensor(model.test(sample), dtype=dtype, requires_grad=True)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        model = ctx.model
        sample = ctx.sample
        dtype = ctx.dtype

        backproper, weight_updater = model._backprop(sample, -grad_outputs[0].detach().numpy())

        gradients = tuple(
            -torch.tensor(sample.get_fact(fact)[0].gradient, dtype=dtype).reshape(fact.weight.shape)
            for fact in ctx.mapping
        )

        trainer = model._strategy.getTrainer()
        trainer.updateWeights(model._strategy.getCurrentModel(), weight_updater)
        trainer.invalidateSample(trainer.getInvalidation(), sample._java_sample)

        return (None, None, None, None, None, *gradients)


class NeuraLogic(nn.Module):
    def __init__(
        self,
        template: Template,
        input_facts: List[Union[BaseRelation, Rule]],
        output_relation: BaseRelation,
        to_logic: Callable,
        settings: Settings | None = None,
        dtype=torch.float32,
    ):
        super(NeuraLogic, self).__init__()

        if settings is None:
            settings = Settings()
        settings.iso_value_compression = False
        settings.chain_pruning = False

        self.dtype = dtype
        self.to_logic = to_logic

        self.model = template.build(settings)
        self.number_format = settings.create_disconnected_proxy().settings_class.superDetailedNumberFormat

        dataset = Dataset(Sample(output_relation, input_facts))
        self.sample = self.model.build_dataset(dataset, learnable_facts=True)[0]
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
            self.dtype,
            *(fact.weight for fact in mapping),
        )
