import os

from neuralogic.utils.data.dataset import Dataset
from neuralogic.core.builder import Backend
from neuralogic.core.settings import Settings

base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..", "..", "dataset")


def XOR(backend: Backend, settings: Settings):
    from neuralogic.core.problem import Problem

    return Problem.build_from_dir(os.path.join(base_path, "simple", "xor", "naive"), backend, settings)


def Trains(backend: Backend, settings: Settings):
    from neuralogic.core.problem import Problem

    return Problem.build_from_dir(os.path.join(base_path, "simple", "trains"), backend, settings)


def XOR_Vectorized(backend: Backend, settings: Settings):
    from neuralogic.core.problem import Problem

    return Problem.build_from_dir(os.path.join(base_path, "simple", "xor", "vectorized"), backend, settings)


def Mutagenesis(backend: Backend, settings: Settings):
    from neuralogic.core.problem import Problem

    return Problem.build_from_dir(os.path.join(base_path, "molecules", "mutagenesis"), backend, settings)
