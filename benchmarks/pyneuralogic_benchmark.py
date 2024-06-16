import time

from torch_geometric.datasets import TUDataset

from benchmarks.helpers import Task
from neuralogic.core import (
    Template,
    Settings,
    R,
    V,
    Transformation,
    Aggregation,
)
from neuralogic.nn.init import Glorot
from neuralogic.nn.loss import CrossEntropy
from neuralogic.dataset import TensorDataset, Data
from neuralogic.optim import Adam


def gcn(activation: Transformation, output_size: int, num_features: int, dim: int = 10):
    template = Template()

    template += R.atom_embed(V.X)[dim, num_features] <= R.node_feature(V.X)

    template += (R.l1_embed(V.X)[dim, dim] <= (R.atom_embed(V.Y), R._edge(V.Y, V.X))) | [Aggregation.SUM]
    template += R.l1_embed / 1 | [Transformation.RELU]

    template += (R.l2_embed(V.X)[dim, dim] <= (R.l1_embed(V.Y), R._edge(V.Y, V.X))) | [Aggregation.SUM]

    template += (R.predict[output_size, dim] <= R.l2_embed(V.X)) | [Aggregation.AVG]
    template += R.predict / 0 | [activation]

    return template


def gin(activation: Transformation, output_size: int, num_features: int, dim: int = 10):
    template = Template()

    template += R.atom_embed(V.X)[dim, num_features] <= R.node_feature(V.X)

    template += (R.l1_embed(V.X) <= (R.atom_embed(V.Y), R._edge(V.Y, V.X))) | [Aggregation.SUM, Transformation.IDENTITY]
    template += R.l1_embed(V.X) <= R.atom_embed(V.X)

    template += (R.l1_mlp_embed(V.X)[dim, dim] <= R.l1_embed(V.X)[dim, dim]) | [Transformation.RELU]
    template += R.l1_mlp_embed / 1 | [Transformation.RELU]

    # --
    template += (R.l2_embed(V.X) <= (R.l1_mlp_embed(V.Y), R._edge(V.Y, V.X))) | [Aggregation.SUM]
    template += R.l2_embed(V.X) <= R.l1_mlp_embed(V.X)

    template += (R.l2_mlp_embed(V.X)[dim, dim] <= R.l2_embed(V.X)[dim, dim]) | [Transformation.RELU]
    template += R.l2_mlp_embed / 1 | [Transformation.RELU]

    # --
    template += (R.l3_embed(V.X) <= (R.l2_mlp_embed(V.Y), R._edge(V.Y, V.X))) | [Aggregation.SUM]
    template += R.l3_embed(V.X) <= R.l2_mlp_embed(V.X)

    template += (R.l3_mlp_embed(V.X)[dim, dim] <= R.l3_embed(V.X)[dim, dim]) | [Transformation.RELU]
    template += R.l3_mlp_embed / 1 | [Transformation.RELU]

    # --
    template += (R.l4_embed(V.X) <= (R.l3_mlp_embed(V.Y), R._edge(V.Y, V.X))) | [Aggregation.SUM]
    template += R.l4_embed(V.X) <= R.l3_mlp_embed(V.X)

    template += (R.l4_mlp_embed(V.X)[dim, dim] <= R.l4_embed(V.X)[dim, dim]) | [Transformation.RELU]
    template += R.l4_mlp_embed / 1 | [Transformation.RELU]

    # --
    template += (R.l5_embed(V.X) <= (R.l4_mlp_embed(V.Y), R._edge(V.Y, V.X))) | [Aggregation.SUM]
    template += R.l5_embed(V.X) <= R.l4_mlp_embed(V.X)

    template += (R.l5_mlp_embed(V.X)[dim, dim] <= R.l5_embed(V.X)[dim, dim]) | [Transformation.RELU]
    template += R.l5_mlp_embed / 1 | [Transformation.RELU]

    template += (R.predict[output_size, dim] <= R.l1_mlp_embed(V.X)) | [Aggregation.AVG]
    template += (R.predict[output_size, dim] <= R.l2_mlp_embed(V.X)) | [Aggregation.AVG]
    template += (R.predict[output_size, dim] <= R.l3_mlp_embed(V.X)) | [Aggregation.AVG]
    template += (R.predict[output_size, dim] <= R.l4_mlp_embed(V.X)) | [Aggregation.AVG]
    template += (R.predict[output_size, dim] <= R.l5_mlp_embed(V.X)) | [Aggregation.AVG]

    template += R.predict / 0 | [activation]

    return template


def gsage(activation: Transformation, output_size: int, num_features: int, dim: int = 10):
    template = Template()

    template += R.atom_embed(V.X)[dim, num_features] <= R.node_feature(V.X)

    template += R.l1_embed(V.X)[dim, dim] <= R.atom_embed(V.X)
    template += R.l1_embed(V.X)[dim, dim] <= (R.atom_embed(V.Y), R._edge(V.Y, V.X)) | [Aggregation.AVG]
    template += R.l1_embed / 1 | [Transformation.RELU]

    template += R.l2_embed(V.X)[dim, dim] <= R.l1_embed(V.X)
    template += (R.l2_embed(V.X)[dim, dim] <= (R.l1_embed(V.Y), R._edge(V.Y, V.X))) | [Aggregation.AVG]

    template += (R.predict[output_size, dim] <= R.l2_embed(V.X)) | [Aggregation.AVG]
    template += R.predict / 0 | [activation]

    return template


def get_model(model):
    if model == "gcn":
        return gcn
    if model == "gsage":
        return gsage
    if model == "gin":
        return gin
    raise NotImplementedError


def evaluate(model, dataset, steps, dataset_loc, dim, task: Task):
    loss_fn = CrossEntropy()
    activation = Transformation.SIGMOID

    settings = Settings(optimizer=Adam(1e-3), error_function=loss_fn, initializer=Glorot())

    ds = TUDataset(root=dataset_loc, name=dataset)

    model = get_model(model)
    model = model(activation=activation, output_size=task.output_size, num_features=ds.num_node_features, dim=dim)
    model = model.build(settings)

    start_time = time.perf_counter()
    dataset = TensorDataset(data=[Data.from_pyg(data)[0] for data in ds], number_of_classes=task.output_size)

    for data in dataset.data:
        data.edge_attr = None

    if task.output_size != 1:
        dataset.number_of_classes = task.output_size
        dataset.one_hot_encoding = True

    built_dataset = model.build_dataset(dataset)

    build_time = time.perf_counter() - start_time
    start_time = time.perf_counter()

    model(built_dataset.samples, train=True, epochs=steps)
    times = [time.perf_counter() - start_time]

    return times, build_time
