from typing import Optional

from neuralogic.data import Dataset
from neuralogic.settings import Settings
from neuralogic.model import Model


class AbstractNeuralogicLayer:
    def __call__(self, sample):
        raise NotImplementedError


class AbstractEvaluator:
    def __init__(self, model: Optional[Model], dataset: Optional[Dataset], settings: Settings):
        if model is None and dataset is None:
            raise NotImplementedError
        if model is not None and dataset is not None:
            raise NotImplementedError

        self.model = model
        self.settings = settings

    def train(self, generator: bool = True):
        pass

    def test(self, generator: bool = True):
        pass
