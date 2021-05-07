import torch.nn.functional as F
import torch

from neuralogic.nn.dgl import NeuraLogicLayer
from neuralogic.nn.base import AbstractEvaluator

from neuralogic.core import Problem
from neuralogic.core.settings import Settings, Optimizer


class DGLEvaluator(AbstractEvaluator):
    trainers = {
        Optimizer.SGD: lambda param, rate: torch.optim.SGD(param, lr=rate),
        Optimizer.ADAM: lambda param, rate: torch.optim.Adam(param, lr=rate),
    }

    def __init__(self, model: Problem, settings: Settings):
        super().__init__(model, settings)

        self.neuralogic_layer = NeuraLogicLayer(None)
        self.dataset = None

    def train(self, generator=False):
        seen_instances = 0
        total_loss = 0
        epochs = self.settings.epochs
        optimizer = self.settings.optimizer

        if optimizer not in DGLEvaluator.trainers:
            raise NotImplementedError

        trainer = DGLEvaluator.trainers[optimizer](self.neuralogic_layer.parameters(), self.settings.learning_rate)

        for _ in range(epochs):
            seen_instances = 0
            total_loss = 0

            for sample in self.dataset.samples:
                self.neuralogic_layer.train()

                label = torch.tensor([sample.target])
                trainer.zero_grad()
                out = self.neuralogic_layer(sample)

                loss = F.mse_loss(out, label)
                loss.backward()
                trainer.step()

                seen_instances += 1
                total_loss += float(loss)
            if generator:
                yield total_loss, seen_instances
        if not generator:
            return total_loss, seen_instances

    def test(self, generator=False):
        def _test():
            for sample in self.dataset.samples:
                yield sample.target, self.neuralogic_layer(sample)

        if generator:
            return _test()
        return list(_test())
