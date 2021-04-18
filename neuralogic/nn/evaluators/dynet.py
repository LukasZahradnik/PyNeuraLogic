from typing import Optional

import dynet as dy

from neuralogic.data import Dataset
from neuralogic.nn.dynet import NeuraLogicLayer
from neuralogic.nn.base import AbstractEvaluator

from neuralogic.model import Model
from neuralogic.settings import Settings, Optimizer
from neuralogic.builder import Backend


class DynetEvaluator(AbstractEvaluator):
    trainers = {
        Optimizer.SGD: lambda param, rate: dy.SimpleSGDTrainer(param, learning_rate=rate),
        Optimizer.ADAM: lambda param, rate: dy.AdamTrainer(param, alpha=rate),
    }

    def __init__(self, model: Optional[Model], dataset: Optional[Dataset], settings: Settings):
        super().__init__(model, dataset, settings)

        if model is not None:
            self.dataset = model.build(Backend.DYNET)
        if dataset is not None:
            self.dataset = dataset
        self.neuralogic_layer = NeuraLogicLayer(self.dataset.weights)

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
