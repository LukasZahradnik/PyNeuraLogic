from typing import Optional

from neuralogic.core.model import Model
from neuralogic.core.settings import Settings
from neuralogic.core import Problem
from neuralogic.utils.data import Dataset


class AbstractNeuralogicLayer:
    def __call__(self, sample):
        raise NotImplementedError


class AbstractEvaluator:
    def __init__(
        self, problem: Optional[Problem], model: Optional[Model], dataset: Optional[Dataset], settings: Settings
    ):
        if model is None and problem is None:
            raise NotImplementedError
        if model is not None and problem is not None:
            raise NotImplementedError

        self.model = model
        self.settings = settings
        self.dataset = dataset

    def train(self, generator: bool = True):
        pass

    def test(self, generator: bool = True):
        pass
