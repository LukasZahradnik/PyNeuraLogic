from typing import Optional, Dict, Union

import torch
import torch.nn.functional

from neuralogic.core.error_function import ErrorFunctionNames
from neuralogic.nn.base import AbstractEvaluator

from neuralogic.core import Template, BuiltDataset, Dataset
from neuralogic.core.settings import Settings
from neuralogic.core.enums import Backend, Optimizer


class TorchEvaluator(AbstractEvaluator):
    trainers = {
        Optimizer.SGD: lambda param, rate: torch.optim.SGD(param, lr=rate),
        Optimizer.ADAM: lambda param, rate: torch.optim.Adam(param, lr=rate),
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

    def train(self, dataset: Optional[Union[Dataset, BuiltDataset]] = None, *, generator: bool = True):
        dataset = self.dataset if dataset is None else self.build_dataset(dataset)

        epochs = self.settings.epochs
        error_function = str(self.settings.error_function)
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

    def test(self, dataset: Optional[Union[Dataset, BuiltDataset]] = None, *, generator: bool = True):
        dataset = self.dataset if dataset is None else self.build_dataset(dataset)

        def _test():
            with torch.no_grad():
                for sample in dataset.samples:
                    graph_output = self.neuralogic_model(sample)

                    if graph_output.size() == (1,):
                        yield sample.target, graph_output[0].item()
                    elif not graph_output.size():
                        yield sample.target, graph_output.item()
                    else:
                        yield sample.target, graph_output

        if generator:
            return _test()
        return list(_test())

    def state_dict(self) -> Dict:
        return self.neuralogic_model.state_dict()

    def load_state_dict(self, state_dict: Dict):
        self.neuralogic_model.load_state_dict(state_dict)
