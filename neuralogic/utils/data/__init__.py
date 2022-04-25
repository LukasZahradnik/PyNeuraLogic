import os

from neuralogic.dataset import FileDataset

base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "datasets")


def XOR():
    from neuralogic.core.template import Template

    template = Template(template_file=os.path.join(base_path, "simple", "xor", "naive", "template.txt"))
    dataset = FileDataset(examples_file=os.path.join(base_path, "simple", "xor", "naive", "trainExamples.txt"))

    return template, dataset


def Family():
    from neuralogic.core.template import Template

    template = Template(template_file=os.path.join(base_path, "simple", "family", "template.txt"))
    dataset = FileDataset(
        examples_file=os.path.join(base_path, "simple", "family", "examples.txt"),
        queries_file=os.path.join(base_path, "simple", "family", "queries.txt"),
    )

    return template, dataset


def Nations():
    from neuralogic.core.template import Template

    template = Template(template_file=os.path.join(base_path, "nations", "template.txt"))
    dataset = FileDataset(
        examples_file=os.path.join(base_path, "nations", "examples.txt"),
        queries_file=os.path.join(base_path, "nations", "queries.txt"),
    )

    return template, dataset


def Trains():
    from neuralogic.core.template import Template

    template = Template(template_file=os.path.join(base_path, "simple", "trains", "template.txt"))
    dataset = FileDataset(
        examples_file=os.path.join(base_path, "simple", "trains", "examples.txt"),
        queries_file=os.path.join(base_path, "simple", "trains", "queries.txt"),
    )

    return template, dataset


def XOR_Vectorized():
    from neuralogic.core.template import Template

    template = Template(template_file=os.path.join(base_path, "simple", "xor", "vectorized", "template.txt"))
    dataset = FileDataset(examples_file=os.path.join(base_path, "simple", "xor", "vectorized", "trainExamples.txt"))

    return template, dataset


def Mutagenesis():
    from neuralogic.core.template import Template

    template = Template(template_file=os.path.join(base_path, "molecules", "mutagenesis", "templates", "template.txt"))
    dataset = FileDataset(
        examples_file=os.path.join(base_path, "molecules", "mutagenesis", "examples.txt"),
        queries_file=os.path.join(base_path, "molecules", "mutagenesis", "queries.txt"),
    )

    return template, dataset
