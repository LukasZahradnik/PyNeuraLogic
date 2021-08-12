from typing import Optional, Dict, Union

import dynet as dy

from neuralogic.nn.base import AbstractEvaluator

from neuralogic.core import Template, BuiltDataset
from neuralogic.core.settings import Settings, Optimizer, ErrorFunction
from neuralogic.core.builder import Backend
from neuralogic.utils.data import Dataset


class DynetEvaluator(AbstractEvaluator):
    trainers = {
        Optimizer.SGD: lambda param, rate: dy.SimpleSGDTrainer(param, learning_rate=rate),
        Optimizer.ADAM: lambda param, rate: dy.AdamTrainer(param, alpha=rate),
    }

    error_functions = {
        ErrorFunction.SQUARED_DIFF: lambda out, target: dy.squared_distance(out, target),
        # ErrorFunction.ABS_DIFF: lambda out, target: dy.abs(out - target),
        # ErrorFunction.CROSSENTROPY: lambda out, target: pass
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
        error_function = ErrorFunction[str(self.settings.error_function)]
        optimizer = Optimizer[str(self.settings.optimizer)]

        if optimizer not in DynetEvaluator.trainers:
            raise NotImplementedError
        if error_function not in DynetEvaluator.error_functions:
            raise NotImplementedError

        trainer = DynetEvaluator.trainers[optimizer](self.neuralogic_model.model, self.settings.learning_rate)
        error_function = DynetEvaluator.error_functions[error_function]

        def _train():
            for _ in range(epochs):
                seen_instances = 0
                total_loss = 0

                dy.renew_cg(immediate_compute=False, check_validity=False)

                for sample in dataset.samples:
                    if isinstance(sample.target, list):
                        label = dy.inputTensor(sample.target)
                    else:
                        label = dy.scalarInput(sample.target)
                    graph_output = self.neuralogic_model(sample)

                    loss = error_function(graph_output, label)

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
                if isinstance(sample.target, list):
                    label = dy.inputTensor(sample.target)
                else:
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
