import os

from neuralogic.dataset import FileDataset

base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "datasets")


def XOR():
    from neuralogic.core.model import Model

    model = Model(model_file=os.path.join(base_path, "simple", "xor", "naive", "template.txt"))
    dataset = FileDataset(examples_file=os.path.join(base_path, "simple", "xor", "naive", "trainExamples.txt"))

    return model, dataset


def Family():
    from neuralogic.core.model import Model

    model = Model(model_file=os.path.join(base_path, "simple", "family", "template.txt"))
    dataset = FileDataset(
        examples_file=os.path.join(base_path, "simple", "family", "examples.txt"),
        queries_file=os.path.join(base_path, "simple", "family", "queries.txt"),
    )

    return model, dataset


def Nations():
    from neuralogic.core.model import Model

    model = Model(model_file=os.path.join(base_path, "nations", "template.txt"))
    dataset = FileDataset(
        examples_file=os.path.join(base_path, "nations", "examples.txt"),
        queries_file=os.path.join(base_path, "nations", "queries.txt"),
    )

    return model, dataset


def Trains():
    from neuralogic.core.model import Model

    model = Model(model_file=os.path.join(base_path, "simple", "trains", "template.txt"))
    dataset = FileDataset(
        examples_file=os.path.join(base_path, "simple", "trains", "examples.txt"),
        queries_file=os.path.join(base_path, "simple", "trains", "queries.txt"),
    )

    return model, dataset


def XOR_Vectorized():
    from neuralogic.core.model import Model

    model = Model(model_file=os.path.join(base_path, "simple", "xor", "vectorized", "template.txt"))
    dataset = FileDataset(examples_file=os.path.join(base_path, "simple", "xor", "vectorized", "trainExamples.txt"))

    return model, dataset


def Mutagenesis():
    from neuralogic.core.model import Model

    model = Model(model_file=os.path.join(base_path, "molecules", "mutagenesis", "templates", "template.txt"))
    dataset = FileDataset(
        examples_file=os.path.join(base_path, "molecules", "mutagenesis", "examples.txt"),
        queries_file=os.path.join(base_path, "molecules", "mutagenesis", "queries.txt"),
    )

    return model, dataset
