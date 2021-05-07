import os

from neuralogic.utils.data.dataset import Dataset
from neuralogic.core.settings import Settings

base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "datasets")


def XOR(settings: Settings):
    from neuralogic.core.problem import Problem

    return Problem(
        settings,
        template_file=os.path.join(base_path, "simple", "xor", "naive", "template.txt"),
        examples_file=os.path.join(base_path, "simple", "xor", "naive", "trainExamples.txt"),
    )


def Trains(settings: Settings):
    from neuralogic.core.problem import Problem

    return Problem(
        settings,
        template_file=os.path.join(base_path, "simple", "trains", "template.txt"),
        examples_file=os.path.join(base_path, "simple", "trains", "examples.txt"),
        queries_file=os.path.join(base_path, "simple", "trains", "queries.txt"),
    )


def XOR_Vectorized(settings: Settings):
    from neuralogic.core.problem import Problem

    return Problem(
        settings,
        template_file=os.path.join(base_path, "simple", "xor", "vectorized", "template.txt"),
        examples_file=os.path.join(base_path, "simple", "xor", "vectorized", "trainExamples.txt"),
    )


def Mutagenesis(settings: Settings):
    from neuralogic.core.problem import Problem

    return Problem(
        settings,
        template_file=os.path.join(base_path, "molecules", "mutagenesis", "templates", "template.txt"),
        examples_file=os.path.join(base_path, "molecules", "mutagenesis", "examples.txt"),
        queries_file=os.path.join(base_path, "molecules", "mutagenesis", "queries.txt"),
    )
