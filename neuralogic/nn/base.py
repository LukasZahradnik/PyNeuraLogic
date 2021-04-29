from typing import Optional

from neuralogic.core.settings import Settings
from neuralogic.core import Problem
from neuralogic.utils.data import Dataset


class AbstractNeuralogicLayer:
    def __call__(self, sample):
        raise NotImplementedError


class AbstractEvaluator:
    def __init__(self, problem: Optional[Problem], model: Optional, dataset: Optional[Dataset], settings: Settings):
        if model is None and problem is None:
            raise NotImplementedError
        if model is not None and problem is not None:
            raise NotImplementedError

        self.settings = settings
        self.dataset = dataset
        self.neuralogic_layer = model

    def train(self, generator: bool = True):
        pass

    def test(self, generator: bool = True):
        pass
