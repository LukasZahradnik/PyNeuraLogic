from typing import Optional, Union

import torch.nn.functional as F
import torch

from neuralogic.core.error_function import ErrorFunctionNames
from neuralogic.nn.base import AbstractEvaluator

from neuralogic.core import Template, BuiltDataset, Backend, Settings, Optimizer, Dataset


class PyGEvaluator(AbstractEvaluator):
    trainers = {
        Optimizer.SGD: lambda param, rate: torch.optim.SGD(param, lr=rate),
        Optimizer.ADAM: lambda param, rate: torch.optim.Adam(param, lr=rate),
    }

    error_functions = {
        str(ErrorFunctionNames.MSE): F.mse_loss,
        str(ErrorFunctionNames.CROSSENTROPY): F.cross_entropy,
    }

    def __init__(self, template: Template, settings: Settings):
        super().__init__(Backend.PYG, template, settings)

    def train(self, dataset: Optional[Union[Dataset, BuiltDataset]] = None, *, generator: bool = True):
        # dataset = self.dataset if dataset is None else self.build_dataset(dataset)

        epochs = self.settings.epochs
        error_function = str(self.settings.error_function)
        optimizer = Optimizer[str(self.settings.optimizer)]

        if optimizer not in PyGEvaluator.trainers:
            raise NotImplementedError
        if error_function not in PyGEvaluator.error_functions:
            raise NotImplementedError

        trainer = PyGEvaluator.trainers[optimizer](
            self.neuralogic_model.module_list.parameters(),
            self.settings.learning_rate,
        )
        error_function = PyGEvaluator.error_functions[error_function]

        def _train():
            for _ in range(epochs):
                seen_instances = 0
                total_loss = 0

                for data in dataset.data:
                    self.neuralogic_model.train()
                    trainer.zero_grad()

                    out = self.neuralogic_model(x=data.x, edge_index=data.edge_index)

                    if data.y_mask is None:
                        loss = error_function(out, data.y)
                    else:
                        loss = error_function(out[data.y_mask], data.y[data.y_mask])
                    loss.backward()
                    trainer.step()

                    seen_instances += 1
                    total_loss += float(loss)
                yield total_loss, seen_instances

        if generator:
            return _train()

        stats = 0, 0
        for stats in _train():
            pass
        return stats

    def test(self, dataset: Optional[Union[Dataset, BuiltDataset]] = None, *, generator: bool = True):
        self.neuralogic_model.train(mode=False)

        # dataset = self.dataset if dataset is None else self.build_dataset(dataset)

        def _test():
            for data in dataset.data:
                self.neuralogic_model.train(mode=False)
                out = self.neuralogic_model(x=data.x, edge_index=data.edge_index)
                results = (out[data.y_mask], data.y[data.y_mask])

                yield results

        if generator:
            return _test()
        return list(_test())
