import os.path as osp

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from neuralogic.utils.templates import TemplateList, GCNConv
from neuralogic.core import Template, Backend, Activation, Settings, Optimizer
from neuralogic.utils.data import Data, Dataset
from neuralogic.nn import get_evaluator


path = osp.join("..", "data", "Cora")
dataset = Planetoid(path, "Cora", transform=T.NormalizeFeatures())

[train_data, test_data, _] = Data.from_pyg(dataset[0])

train_dataset = Dataset(data=[train_data])
test_dataset = Dataset(data=[test_data])


torch.manual_seed(123)

template_list = TemplateList(
    [
        GCNConv(in_channels=dataset.num_features, out_channels=16, activation=Activation.RELU),
        GCNConv(in_channels=16, out_channels=dataset.num_classes, activation=Activation.SIGMOID),
    ]
)

template = Template(module_list=template_list)

settings = Settings(epochs=1000, optimizer=Optimizer.ADAM, learning_rate=0.01)
evaluator = get_evaluator(Backend.PYG, template, settings, native_backend_models=True)

for epoch, (total_loss, seen_instances) in enumerate(evaluator.train(train_dataset)):
    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}, total loss: {total_loss}, instances: {seen_instances}, average loss {total_loss / seen_instances}"
        )


for epoch, ok in enumerate(evaluator.test(test_dataset)):
    print(ok)
