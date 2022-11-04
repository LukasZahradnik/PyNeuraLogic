from typing import Optional, Dict, Union

from neuralogic.core import Template, BuiltDataset
from neuralogic.nn.base import AbstractEvaluator
from neuralogic.core.settings import Settings

from neuralogic.dataset.base import BaseDataset


class JavaEvaluator(AbstractEvaluator):
    def __init__(
        self,
        template: Template,
        settings: Settings,
    ):
        super().__init__(template, settings)

    def train(
        self,
        dataset: Optional[Union[BaseDataset, BuiltDataset]] = None,
        *,
        generator: bool = True,
        epochs: int = None,
    ):
        dataset = self.build_dataset(dataset)

        if epochs is None:
            epochs = self.settings.epochs

        def _train():
            for _ in range(epochs):
                results, total_len = self.neuralogic_model(dataset, True)
                yield sum(result[2] for result in results), total_len

        if generator:
            return _train()

        results, total_len = self.neuralogic_model(dataset, True, epochs=epochs)

        return sum(result[2] for result in results), total_len

    def test(
        self, dataset: Optional[Union[BaseDataset, BuiltDataset]] = None, *, generator: bool = True, batch_size: int = 1
    ):
        dataset = self.build_dataset(dataset)

        def _test():
            for sample in dataset.samples:
                yield self.neuralogic_model(sample, False)

        if generator:
            return _test()
        return self.neuralogic_model(dataset, False)

    def state_dict(self) -> Dict:
        return self.neuralogic_model.state_dict()

    def load_state_dict(self, state_dict: Dict):
        self.neuralogic_model.load_state_dict(state_dict)
