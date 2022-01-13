import time

from torch_geometric.datasets import TUDataset

from neuralogic.core.dataset import Dataset, Data
from neuralogic.core import Template, Backend, Settings, Optimizer, ErrorFunction, R, V, Activation, Aggregation


def gcn(num_features: int, dim: int = 10):
    template = Template()

    template += (R.atom_embed(V.X)[dim, num_features] <= R.node_feature(V.X)) | [Activation.IDENTITY]
    template += R.atom_embed / 1 | [Activation.IDENTITY]

    template += (R.l1_embed(V.X)[dim, dim] <= (R.atom_embed(V.Y), R._edge(V.X, V.Y))) | [
        Aggregation.SUM,
        Activation.IDENTITY,
    ]
    template += R.l1_embed / 1 | [Activation.RELU]

    template += (R.l2_embed(V.X)[dim, dim] <= (R.l1_embed(V.Y), R._edge(V.X, V.Y))) | [
        Aggregation.SUM,
        Activation.IDENTITY,
    ]
    template += R.l2_embed / 1 | [Activation.IDENTITY]

    template += (R.predict[1, dim] <= R.l2_embed(V.X)) | [Aggregation.AVG, Activation.IDENTITY]
    template += R.predict / 0 | [Activation.SIGMOID]

    return template


def gin(num_features: int, dim: int = 10):
    template = Template()

    template += (R.atom_embed(V.X)[dim, num_features] <= R.node_feature(V.X)) | [Activation.IDENTITY]
    template += R.atom_embed / 1 | [Activation.IDENTITY]

    template += (R.l1_embed(V.X) <= (R.atom_embed(V.Y), R._edge(V.X, V.Y))) | [Aggregation.SUM, Activation.IDENTITY]
    template += (R.l1_embed(V.X) <= R.atom_embed(V.X)) | [Activation.IDENTITY]
    template += R.l1_embed / 1 | [Activation.IDENTITY]

    template += (R.l1_mlp_embed(V.X)[dim, dim] <= R.l1_embed(V.X)[dim, dim]) | [Activation.RELU]
    template += R.l1_mlp_embed / 1 | [Activation.RELU]

    # --
    template += (R.l2_embed(V.X) <= (R.l1_mlp_embed(V.Y), R._edge(V.X, V.Y))) | [Aggregation.SUM, Activation.IDENTITY]
    template += (R.l2_embed(V.X) <= R.l1_mlp_embed(V.X)) | [Activation.IDENTITY]
    template += R.l2_embed / 1 | [Activation.IDENTITY]

    template += (R.l2_mlp_embed(V.X)[dim, dim] <= R.l2_embed(V.X)[dim, dim]) | [Activation.RELU]
    template += R.l2_mlp_embed / 1 | [Activation.RELU]

    # --
    template += (R.l3_embed(V.X) <= (R.l2_mlp_embed(V.Y), R._edge(V.X, V.Y))) | [Aggregation.SUM, Activation.IDENTITY]
    template += (R.l3_embed(V.X) <= R.l2_mlp_embed(V.X)) | [Activation.IDENTITY]
    template += R.l3_embed / 1 | [Activation.IDENTITY]

    template += (R.l3_mlp_embed(V.X)[dim, dim] <= R.l3_embed(V.X)[dim, dim]) | [Activation.RELU]
    template += R.l3_mlp_embed / 1 | [Activation.RELU]

    # --
    template += (R.l4_embed(V.X) <= (R.l3_mlp_embed(V.Y), R._edge(V.X, V.Y))) | [Aggregation.SUM, Activation.IDENTITY]
    template += (R.l4_embed(V.X) <= R.l3_mlp_embed(V.X)) | [Activation.IDENTITY]
    template += R.l4_embed / 1 | [Activation.IDENTITY]

    template += (R.l4_mlp_embed(V.X)[dim, dim] <= R.l4_embed(V.X)[dim, dim]) | [Activation.RELU]
    template += R.l4_mlp_embed / 1 | [Activation.RELU]

    # --
    template += (R.l5_embed(V.X) <= (R.l4_mlp_embed(V.Y), R._edge(V.X, V.Y))) | [Aggregation.SUM, Activation.IDENTITY]
    template += (R.l5_embed(V.X) <= R.l4_mlp_embed(V.X)) | [Activation.IDENTITY]
    template += R.l5_embed / 1 | [Activation.IDENTITY]

    template += (R.l5_mlp_embed(V.X)[dim, dim] <= R.l5_embed(V.X)[dim, dim]) | [Activation.RELU]
    template += R.l5_mlp_embed / 1 | [Activation.RELU]

    template += (R.predict[1, dim] <= R.l1_mlp_embed(V.X)) | [Aggregation.AVG, Activation.IDENTITY]
    template += (R.predict[1, dim] <= R.l2_mlp_embed(V.X)) | [Aggregation.AVG, Activation.IDENTITY]
    template += (R.predict[1, dim] <= R.l3_mlp_embed(V.X)) | [Aggregation.AVG, Activation.IDENTITY]
    template += (R.predict[1, dim] <= R.l4_mlp_embed(V.X)) | [Aggregation.AVG, Activation.IDENTITY]
    template += (R.predict[1, dim] <= R.l5_mlp_embed(V.X)) | [Aggregation.AVG, Activation.IDENTITY]

    template += R.predict / 0 | [Activation.SIGMOID]

    return template


def gsage(num_features: int, dim: int = 10):
    template = Template()

    template += (R.atom_embed(V.X)[dim, num_features] <= R.node_feature(V.X)) | [Activation.IDENTITY]
    template += R.atom_embed / 1 | [Activation.IDENTITY]

    template += (R.l1_embed(V.X)[dim, dim] <= R.atom_embed(V.X)) | [Activation.IDENTITY]
    template += (R.l1_embed(V.X)[dim, dim] <= (R.atom_embed(V.Y), R._edge(V.X, V.Y))) | [
        Aggregation.AVG,
        Activation.IDENTITY,
    ]
    template += R.l1_embed / 1 | [Activation.RELU]

    template += (R.l2_embed(V.X)[dim, dim] <= R.l1_embed(V.X)) | [Activation.IDENTITY]
    template += (R.l2_embed(V.X)[dim, dim] <= (R.l1_embed(V.Y), R._edge(V.X, V.Y))) | [
        Aggregation.AVG,
        Activation.IDENTITY,
    ]
    template += R.l2_embed / 1 | [Activation.IDENTITY]

    template += (R.predict[1, dim] <= R.l2_embed(V.X)) | [Aggregation.AVG, Activation.IDENTITY]
    template += R.predict / 0 | [Activation.SIGMOID]

    return template


def get_model(model):
    if model == "gcn":
        return gcn
    if model == "gsage":
        return gsage
    if model == "gin":
        return gin
    raise NotImplementedError


def evaluate(model, dataset, steps, dataset_loc, dim):
    settings = Settings(optimizer=Optimizer.ADAM, error_function=ErrorFunction.CROSSENTROPY, learning_rate=1e-3)

    ds = TUDataset(root=dataset_loc, name=dataset)
    model = get_model(model)(num_features=ds.num_node_features, dim=dim).build(Backend.JAVA, settings)

    dataset = Dataset(data=[Data.from_pyg(data)[0] for data in ds])
    built_dataset = model.build_dataset(dataset)

    start_time = time.perf_counter()
    model(built_dataset.samples, train=True, epochs=steps)
    times = [time.perf_counter() - start_time]

    return times
