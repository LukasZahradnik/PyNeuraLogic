from typing import Optional, Dict, Union

import torch
import torch.nn.functional

from neuralogic.nn.loss import ErrorFunctionNames
from neuralogic.dataset.base import BaseDataset
from neuralogic.nn.base import AbstractEvaluator

from neuralogic.core import Template, BuiltDataset
from neuralogic.core.settings import Settings
from neuralogic.core.enums import Backend


class TorchEvaluator(AbstractEvaluator):
    trainers = {
        "SGD": lambda param, optimizer: torch.optim.SGD(param, lr=optimizer.lr),
        "Adam": lambda param, optimizer: torch.optim.Adam(
            param, lr=optimizer.lr, betas=optimizer.betas, eps=optimizer.eps
        ),
    }

    error_functions = {
        str(ErrorFunctionNames.MSE): torch.nn.MSELoss(),
        str(ErrorFunctionNames.SOFTENTROPY): torch.nn.CrossEntropyLoss(),
        # ErrorFunction.ABS_DIFF: lambda out, target: dy.abs(out - target),
        # ErrorFunction.CROSSENTROPY: lambda out, target: pass
    }

    def __init__(
        self,
        template: Template,
        settings: Settings,
    ):
        super().__init__(Backend.TORCH, template, settings)

    def train(self, dataset: Optional[Union[BaseDataset, BuiltDataset]] = None, *, generator: bool = True):
        dataset = self.dataset if dataset is None else self.build_dataset(dataset)

        epochs = self.settings.epochs
        error_function = str(self.settings.error_function)
        optimizer = self.settings.optimizer

        if optimizer.name() not in TorchEvaluator.trainers:
            raise NotImplementedError
        if error_function not in TorchEvaluator.error_functions:
            raise NotImplementedError

        trainer = TorchEvaluator.trainers[optimizer.name()](self.neuralogic_model.model, optimizer)
        error_function = TorchEvaluator.error_functions[error_function]

        def _train():
            for _ in range(epochs):
                seen_instances = 0
                total_loss = 0
                for sample in dataset.samples:
                    trainer.zero_grad(set_to_none=True)

                    if isinstance(sample.target, (int, float)):
                        label = torch.tensor([sample.target], dtype=torch.float64, requires_grad=False)
                    else:
                        label = torch.tensor(sample.target, dtype=torch.float64, requires_grad=False)

                    graph_output = self.neuralogic_model(sample)
                    loss = error_function(graph_output, label)

                    try:
                        loss.backward()
                    except RuntimeError:
                        pass
                    trainer.step()

                    total_loss += loss.item()
                    seen_instances += 1
                yield total_loss, seen_instances

        if generator:
            return _train()

        stats = 0, 0
        for stats in _train():
            pass
        return stats

    def test(self, dataset: Optional[Union[BaseDataset, BuiltDataset]] = None, *, generator: bool = True):
        dataset = self.dataset if dataset is None else self.build_dataset(dataset)

        def _test():
            with torch.no_grad():
                for sample in dataset.samples:
                    graph_output = self.neuralogic_model(sample)

                    if graph_output.size() == (1,):
                        yield graph_output[0].item()
                    elif not graph_output.size():
                        yield graph_output.item()
                    else:
                        yield graph_output

        if generator:
            return _test()
        return list(_test())

    def state_dict(self) -> Dict:
        return self.neuralogic_model.state_dict()

    def load_state_dict(self, state_dict: Dict):
        self.neuralogic_model.load_state_dict(state_dict)
