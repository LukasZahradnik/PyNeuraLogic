import os
from typing import Tuple

from neuralogic.core.builder import Backend
from neuralogic.core.model import Model
from neuralogic.core.settings import Settings


class Dataset:
    def __init__(self, samples):
        self.__samples = samples
        self.__len = len(samples)

    def __len__(self):
        return self.__len

    def __getitem__(self, item):
        return self.__samples[item]

    @property
    def samples(self):
        return self.__samples


base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..", "..", "dataset")


def XOR(backend: Backend, settings: Settings) -> Tuple[Model, Dataset]:
    from neuralogic.core.problem import Problem

    return Problem.build_from_dir(os.path.join(base_path, "simple", "xor", "naive"), backend, settings)


def Trains(backend: Backend, settings: Settings) -> Tuple[Model, Dataset]:
    from neuralogic.core.problem import Problem

    return Problem.build_from_dir(os.path.join(base_path, "simple", "trains"), backend, settings)


def XOR_Vectorized(settings: Settings, backend: Backend) -> Tuple[Model, Dataset]:
    from neuralogic.core.problem import Problem

    return Problem.build_from_dir(os.path.join(base_path, "simple", "xor", "vectorized"), backend, settings)


def Mutagenesis(settings: Settings, backend: Backend) -> Tuple[Model, Dataset]:
    from neuralogic.core.problem import Problem

    return Problem.build_from_dir(os.path.join(base_path, "molecules", "mutagenesis"), backend, settings)
