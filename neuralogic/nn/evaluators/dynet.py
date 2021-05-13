from typing import Optional, Dict, Union

import dynet as dy

from neuralogic.nn.dynet import NeuraLogic
from neuralogic.nn.base import AbstractEvaluator

from neuralogic.core import Template, BuiltDataset
from neuralogic.core.settings import Settings, Optimizer
from neuralogic.core.builder import Backend
from neuralogic.utils.data import Dataset


class DynetEvaluator(AbstractEvaluator):
    trainers = {
        Optimizer.SGD: lambda param, rate: dy.SimpleSGDTrainer(param, learning_rate=rate),
        Optimizer.ADAM: lambda param, rate: dy.AdamTrainer(param, alpha=rate),
    }

    def __init__(
        self,
        template: Template,
        settings: Settings,
    ):
        super().__init__(Backend.DYNET, template, settings)

    def train(self, dataset: Optional[Union[Dataset, BuiltDataset]] = None, *, generator: bool = True):
        dataset = self.dataset if dataset is None else self.build_dataset(dataset)

        epochs = self.settings.epochs
        optimizer = Optimizer[str(self.settings.optimizer)]

        if optimizer not in DynetEvaluator.trainers:
            raise NotImplementedError

        trainer = DynetEvaluator.trainers[optimizer](self.neuralogic_model.model, self.settings.learning_rate)

        def _train():
            for _ in range(epochs):
                seen_instances = 0
                total_loss = 0

                dy.renew_cg(immediate_compute=False, check_validity=False)

                for sample in dataset.samples:
                    label = dy.scalarInput(sample.target)
                    graph_output = self.neuralogic_model(sample)
                    loss = dy.squared_distance(graph_output, label)

                    total_loss += loss.value()
                    loss.backward()
                    trainer.update()
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
            for sample in dataset.samples:
                dy.renew_cg(immediate_compute=False, check_validity=False)

                graph_output = self.neuralogic_model(sample)
                label = dy.scalarInput(sample.target)

                results = (label.value(), graph_output.value())
                yield results

        if generator:
            return _test()
        return list(_test())

    def state_dict(self) -> Dict:
        return self.neuralogic_model.state_dict()

    def load_state_dict(self, state_dict: Dict):
        self.neuralogic_model.load_state_dict(state_dict)
