from typing import Optional, Dict, Union

import torch

from neuralogic.nn.base import AbstractEvaluator

from neuralogic.core import Template, BuiltDataset, Dataset
from neuralogic.core.settings import Settings
from neuralogic.core.enums import Backend, Optimizer, ErrorFunction


class TorchEvaluator(AbstractEvaluator):
    trainers = {
        Optimizer.SGD: lambda param, rate: torch.optim.SGD(param, lr=rate),
        Optimizer.ADAM: lambda param, rate: torch.optim.Adam(param, lr=rate),
    }

    error_functions = {
        ErrorFunction.SQUARED_DIFF: torch.nn.MSELoss()
        # ErrorFunction.ABS_DIFF: lambda out, target: dy.abs(out - target),
        # ErrorFunction.CROSSENTROPY: lambda out, target: pass
    }

    def __init__(
        self,
        template: Template,
        settings: Settings,
    ):
        super().__init__(Backend.TORCH, template, settings)

    def train(self, dataset: Optional[Union[Dataset, BuiltDataset]] = None, *, generator: bool = True):
        dataset = self.dataset if dataset is None else self.build_dataset(dataset)

        epochs = self.settings.epochs
        error_function = ErrorFunction[str(self.settings.error_function)]
        optimizer = Optimizer[str(self.settings.optimizer)]

        if optimizer not in TorchEvaluator.trainers:
            raise NotImplementedError
        if error_function not in TorchEvaluator.error_functions:
            raise NotImplementedError

        trainer = TorchEvaluator.trainers[optimizer](self.neuralogic_model.model, self.settings.learning_rate)
        error_function = TorchEvaluator.error_functions[error_function]

        def _train():
            for _ in range(epochs):
                seen_instances = 0
                total_loss = 0

                for sample in dataset.samples:
                    trainer.zero_grad()

                    if isinstance(sample.target, list):
                        label = torch.tensor(sample.target, dtype=torch.float64)
                    else:
                        label = torch.scalar_tensor(sample.target, dtype=torch.float64)

                    graph_output = self.neuralogic_model(sample)
                    loss = error_function(graph_output, label)

                    total_loss += loss.data

                    loss.backward()
                    trainer.step()
                    seen_instances += 1
                yield total_loss, seen_instances

        if generator:
            return _train()

        stats = 0, 0
        for stats in _train():
            pass
        return stats

    def test(self, dataset: Optional[Union[Dataset, BuiltDataset]] = None, *, generator: bool = True):
        dataset = self.dataset if dataset is None else self.build_dataset(dataset)

        def _test():
            with torch.no_grad():
                for sample in dataset.samples:
                    graph_output = self.neuralogic_model(sample)

                    if isinstance(sample.target, list):
                        label = torch.tensor(sample.target)
                    else:
                        label = torch.scalar_tensor(sample.target)

                    results = (label.data, graph_output)
                    yield results

        if generator:
            return _test()
        return list(_test())

    def state_dict(self) -> Dict:
        return self.neuralogic_model.state_dict()

    def load_state_dict(self, state_dict: Dict):
        self.neuralogic_model.load_state_dict(state_dict)
