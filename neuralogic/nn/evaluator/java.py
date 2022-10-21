from typing import Optional, Dict, Union

import jpype

from neuralogic.core import Template, BuiltDataset
from neuralogic.nn.base import AbstractEvaluator
from neuralogic.core.settings import Settings

from neuralogic.dataset.base import BaseDataset


class JavaEvaluator(AbstractEvaluator):
    def __init__(
        self,
        template: Optional[Template],
        settings: Settings,
    ):
        super().__init__(template, settings)

    def set_dataset(self, dataset: Union[BaseDataset, BuiltDataset]):
        super().set_dataset(dataset)
        self.neuralogic_model.set_training_samples(
            jpype.java.util.ArrayList([sample.java_sample for sample in self.dataset.samples])
        )

    def reset_dataset(self, dataset):
        if dataset is None:
            self.neuralogic_model.set_training_samples(jpype.java.util.ArrayList([]))
        else:
            self.neuralogic_model.set_training_samples(
                jpype.java.util.ArrayList([sample.java_sample for sample in dataset.samples])
            )
        self.dataset = dataset

    def train(
        self,
        dataset: Optional[Union[BaseDataset, BuiltDataset]] = None,
        *,
        generator: bool = True,
        epochs: int = None,
        batch_size: int = 1,
    ):
        old_dataset = None

        if dataset is not None:
            old_dataset = self.dataset
            self.set_dataset(dataset)

        if epochs is None:
            epochs = self.settings.epochs

        def _train():
            for _ in range(epochs):
                results, total_len = self.neuralogic_model(None, True, batch_size=batch_size)
                yield sum(result[2] for result in results), total_len
            if dataset is not None:
                self.reset_dataset(old_dataset)

        if generator:
            return _train()

        results, total_len = self.neuralogic_model(None, True, epochs=epochs, batch_size=batch_size)
        if dataset is not None:
            self.reset_dataset(old_dataset)

        return sum(result[2] for result in results), total_len

    def test(
        self, dataset: Optional[Union[BaseDataset, BuiltDataset]] = None, *, generator: bool = True, batch_size: int = 1
    ):
        dataset = self.dataset if dataset is None else self.build_dataset(dataset)

        def _test():
            for sample in dataset.samples:
                yield self.neuralogic_model(sample, False, batch_size=batch_size)

        if generator:
            return _test()
        return self.neuralogic_model(dataset.samples, False, batch_size=batch_size)

    def state_dict(self) -> Dict:
        return self.neuralogic_model.state_dict()

    def load_state_dict(self, state_dict: Dict):
        self.neuralogic_model.load_state_dict(state_dict)
