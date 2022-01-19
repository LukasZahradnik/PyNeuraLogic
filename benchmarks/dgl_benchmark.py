import time

import dgl
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv, GINConv
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.datasets import TUDataset

from benchmarks.helpers import Task


class NetGCN(torch.nn.Module):
    def __init__(self, activation: str, output_size: int, num_features, dim=10):
        super(NetGCN, self).__init__()
        self.conv1 = GraphConv(num_features, dim, norm="none", bias=False, allow_zero_in_degree=True)
        self.conv2 = GraphConv(dim, dim, norm="none", bias=False, allow_zero_in_degree=True)
        self.fc1 = Linear(dim, output_size, bias=False)

        self.activation = getattr(torch, activation)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)

        g.ndata["h"] = x
        x = dgl.mean_nodes(g, "h")
        x = self.fc1(x)

        return self.activation(x)


class NetGraphSage(torch.nn.Module):
    def __init__(self, activation: str, output_size: int, num_features, dim=10):
        super(NetGraphSage, self).__init__()
        self.conv1 = SAGEConv(num_features, dim, aggregator_type="mean", bias=False)
        self.conv2 = SAGEConv(dim, dim, aggregator_type="mean", bias=False)
        self.fc1 = Linear(dim, output_size, bias=False)

        self.activation = getattr(torch, activation)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)

        g.ndata["h"] = x
        x = dgl.mean_nodes(g, "h")

        x = self.fc1(x)

        return self.activation(x)


class NetGIN(torch.nn.Module):
    def __init__(self, activation: str, output_size: int, num_features, dim=10):
        super(NetGIN, self).__init__()

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1, "sum")

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2, "sum")

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3, "sum")

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4, "sum")

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5, "sum")

        self.l1 = Linear(dim, output_size, bias=False)
        self.l2 = Linear(dim, output_size, bias=False)
        self.l3 = Linear(dim, output_size, bias=False)
        self.l4 = Linear(dim, output_size, bias=False)
        self.l5 = Linear(dim, output_size, bias=False)

        self.activation = getattr(torch, activation)

    def forward(self, g, h):
        x1 = F.relu(self.conv1(g, h))
        x2 = F.relu(self.conv2(g, x1))
        x3 = F.relu(self.conv3(g, x2))
        x4 = F.relu(self.conv4(g, x3))
        x5 = F.relu(self.conv5(g, x4))

        g.ndata["h1"] = x1
        m1 = dgl.mean_nodes(g, "h1")
        g.ndata["h2"] = x2
        m2 = dgl.mean_nodes(g, "h2")
        g.ndata["h3"] = x3
        m3 = dgl.mean_nodes(g, "h3")
        g.ndata["h4"] = x4
        m4 = dgl.mean_nodes(g, "h4")
        g.ndata["h5"] = x5
        m5 = dgl.mean_nodes(g, "h5")

        sum_ = torch.stack([self.l1(m1), self.l2(m2), self.l3(m3), self.l4(m4), self.l5(m5)], dim=0)

        x = torch.sum(sum_, dim=0)

        return self.activation(x)


def get_model(model):
    if model == "gcn":
        return NetGCN
    if model == "gsage":
        return NetGraphSage
    if model == "gin":
        return NetGIN
    raise NotImplementedError


def evaluate(model, dataset, steps, dataset_loc, dim, task: Task):
    ds = TUDataset(root=dataset_loc, name=dataset)

    model = get_model(model)
    model = model(activation=task.activation, output_size=task.output_size, num_features=ds.num_node_features, dim=dim)

    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = F.binary_cross_entropy if task.output_size == 1 else F.cross_entropy

    dataset = []
    for graph in ds:
        u, v = graph["edge_index"]
        g = dgl.graph((u, v), num_nodes=graph.num_nodes)

        g.ndata["x"] = graph["x"]
        g.y = graph.y

        if task.output_size == 1:
            g.y = g.y.float()
        dataset.append(g)

    times = []
    for _ in range(steps):
        model.train()

        tm = 0
        ls = 0
        for data in dataset:
            t = time.perf_counter()
            data = data.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(data, data.ndata["x"])

            loss = loss_fn(output, data.y)

            loss.backward()
            optimizer.step()

            tm += time.perf_counter() - t
            ls += loss.item()
        print(tm)
        times.append(tm)
    return times, 0
