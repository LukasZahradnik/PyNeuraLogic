import os
from typing import Optional

from neuralogic.utils.data.dataset import Dataset, Data
from neuralogic.core.settings import Settings

base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "datasets")


def XOR(settings: Optional[Settings] = None):
    from neuralogic.core.template import Template

    template = Template(settings, template_file=os.path.join(base_path, "simple", "xor", "naive", "template.txt"))
    dataset = Dataset(examples_file=os.path.join(base_path, "simple", "xor", "naive", "trainExamples.txt"))

    return template, dataset


def Trains(settings: Optional[Settings] = None):
    from neuralogic.core.template import Template

    template = Template(settings, template_file=os.path.join(base_path, "simple", "trains", "template.txt"))
    dataset = Dataset(
        examples_file=os.path.join(base_path, "simple", "trains", "examples.txt"),
        queries_file=os.path.join(base_path, "simple", "trains", "queries.txt"),
    )

    return template, dataset


def XOR_Vectorized(settings: Optional[Settings] = None):
    from neuralogic.core.template import Template

    template = Template(settings, template_file=os.path.join(base_path, "simple", "xor", "vectorized", "template.txt"))
    dataset = Dataset(examples_file=os.path.join(base_path, "simple", "xor", "vectorized", "trainExamples.txt"))

    return template, dataset


def Mutagenesis(settings: Optional[Settings] = None):
    from neuralogic.core.template import Template

    template = Template(
        settings,
        template_file=os.path.join(base_path, "molecules", "mutagenesis", "templates", "template.txt"),
    )

    dataset = Dataset(
        examples_file=os.path.join(base_path, "molecules", "mutagenesis", "examples.txt"),
        queries_file=os.path.join(base_path, "molecules", "mutagenesis", "queries.txt"),
    )

    return template, dataset
