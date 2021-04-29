from typing import Optional, Dict

import dynet as dy

from neuralogic.nn.dynet import NeuraLogicLayer
from neuralogic.nn.base import AbstractEvaluator

from neuralogic.core import Problem
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
        problem: Optional[Problem],
        model: Optional[NeuraLogicLayer],
        dataset: Optional[Dataset],
        settings: Settings,
    ):
        super().__init__(problem, model, dataset, settings)

        if problem is not None:
            model, dataset = problem.build(Backend.DYNET)
            self.dataset = dataset
        self.neuralogic_layer = model

    def train(self, generator: bool = True):
        epochs = self.settings.epochs
        optimizer = Optimizer[str(self.settings.optimizer)]

        if optimizer not in DynetEvaluator.trainers:
            raise NotImplementedError

        trainer = DynetEvaluator.trainers[optimizer](self.neuralogic_layer.model, self.settings.learning_rate)

        def _train():
            for _ in range(epochs):
                seen_instances = 0
                total_loss = 0

                dy.renew_cg(immediate_compute=False, check_validity=False)

                for sample in self.dataset.samples:
                    label = dy.scalarInput(sample.target)
                    graph_output = self.neuralogic_layer(sample)
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

    def test(self, generator: bool = True):
        def _test():
            for sample in self.dataset.samples:
                dy.renew_cg(immediate_compute=False, check_validity=False)

                graph_output = self.neuralogic_layer(sample)
                label = dy.scalarInput(sample.target)

                results = (label.value(), graph_output.value())
                yield results

        if generator:
            return _test()
        return list(_test())

    def state_dict(self) -> Dict:
        return self.neuralogic_layer.state_dict()

    def load_state_dict(self, state_dict: Dict):
        self.neuralogic_layer.load_state_dict(state_dict)
