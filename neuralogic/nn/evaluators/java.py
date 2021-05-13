from typing import Optional, Dict, Union

from py4j.java_collections import ListConverter

from neuralogic import get_gateway
from neuralogic.core.builder import Backend
from neuralogic.core import Template, BuiltDataset
from neuralogic.nn.base import AbstractEvaluator
from neuralogic.core.settings import Settings
from neuralogic.utils.data import Dataset


class JavaEvaluator(AbstractEvaluator):
    def __init__(
        self,
        problem: Optional[Template],
        settings: Settings,
    ):
        super().__init__(Backend.JAVA, problem, settings)

    def set_dataset(self, dataset: Union[Dataset, BuiltDataset]):
        super().set_dataset(dataset)
        self.neuralogic_model.set_training_samples(self.dataset.samples)

    def reset_dataset(self, dataset):
        if dataset is None:
            empty_list = ListConverter().convert([], get_gateway()._gateway_client)
            self.neuralogic_model.set_training_samples(empty_list)
        else:
            self.neuralogic_model.set_training_samples(dataset.samples)
        self.dataset = dataset

    def train(
        self, dataset: Optional[Union[Dataset, BuiltDataset]] = None, *, generator: bool = True, epochs: int = None
    ):
        old_dataset = None

        if dataset is not None:
            old_dataset = self.dataset
            self.set_dataset(dataset)

        if epochs is None:
            epochs = self.settings.epochs

        def _train():
            for _ in range(epochs):
                result = self.neuralogic_model(None, True)
                yield result
            if dataset is not None:
                self.reset_dataset(old_dataset)

        if generator:
            return _train()

        results = self.neuralogic_model(None, True, epochs=epochs)
        if dataset is not None:
            self.reset_dataset(old_dataset)

        return results

    def test(self, dataset: Optional[Union[Dataset, BuiltDataset]] = None, *, generator: bool = True):
        dataset = self.dataset if dataset is None else self.build_dataset(dataset)

        def _test():
            for sample in dataset.samples:
                result = self.neuralogic_model(sample, False)
                yield result.target(), result.output()

        if generator:
            return _test()
        return self.neuralogic_model(dataset.samples, False)

    def state_dict(self) -> Dict:
        return self.neuralogic_model.state_dict()

    def load_state_dict(self, state_dict: Dict):
        self.neuralogic_model.load_state_dict(state_dict)
