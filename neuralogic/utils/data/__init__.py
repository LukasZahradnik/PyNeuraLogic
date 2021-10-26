import os

from neuralogic.core.dataset import Dataset


base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "datasets")


def XOR():
    from neuralogic.core.template import Template

    template = Template(template_file=os.path.join(base_path, "simple", "xor", "naive", "template.txt"))
    dataset = Dataset(examples_file=os.path.join(base_path, "simple", "xor", "naive", "trainExamples.txt"))

    return template, dataset


def Trains():
    from neuralogic.core.template import Template

    template = Template(template_file=os.path.join(base_path, "simple", "trains", "template.txt"))
    dataset = Dataset(
        examples_file=os.path.join(base_path, "simple", "trains", "examples.txt"),
        queries_file=os.path.join(base_path, "simple", "trains", "queries.txt"),
    )

    return template, dataset


def XOR_Vectorized():
    from neuralogic.core.template import Template

    template = Template(template_file=os.path.join(base_path, "simple", "xor", "vectorized", "template.txt"))
    dataset = Dataset(examples_file=os.path.join(base_path, "simple", "xor", "vectorized", "trainExamples.txt"))

    return template, dataset


def Mutagenesis():
    from neuralogic.core.template import Template

    template = Template(template_file=os.path.join(base_path, "molecules", "mutagenesis", "templates", "template.txt"))
    dataset = Dataset(
        examples_file=os.path.join(base_path, "molecules", "mutagenesis", "examples.txt"),
        queries_file=os.path.join(base_path, "molecules", "mutagenesis", "queries.txt"),
    )

    return template, dataset
