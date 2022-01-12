import time

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_mean_pool, GINConv, SAGEConv


class NetGCN(torch.nn.Module):
    def __init__(self, num_features: int, dim: int = 10):
        super(NetGCN, self).__init__()

        self.conv1 = GCNConv(num_features, dim, normalize=False, cached=False, bias=False)
        self.conv2 = GCNConv(dim, dim, normalize=False, cached=False, bias=False)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

        self.fc1 = Linear(dim, 1, bias=False)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        return torch.sigmoid(x)


class NetGraphSage(torch.nn.Module):
    def __init__(self, num_features: int, dim: int = 10):
        super(NetGraphSage, self).__init__()
        self.conv1 = SAGEConv(num_features, dim, normalize=False, bias=False)
        self.conv2 = SAGEConv(dim, dim, normalize=False, bias=False)

        self.fc1 = Linear(dim, 1, bias=False)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        return torch.sigmoid(x)


class NetGIN(torch.nn.Module):
    def __init__(self, num_features: int, dim: int = 10):
        super(NetGIN, self).__init__()

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)

        self.l1 = Linear(dim, 1, bias=False)
        self.l2 = Linear(dim, 1, bias=False)
        self.l3 = Linear(dim, 1, bias=False)
        self.l4 = Linear(dim, 1, bias=False)
        self.l5 = Linear(dim, 1, bias=False)

    def forward(self, x, edge_index, batch):
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2, edge_index))
        x4 = F.relu(self.conv4(x3, edge_index))
        x5 = F.relu(self.conv5(x4, edge_index))

        m1 = global_mean_pool(x1, batch)
        m2 = global_mean_pool(x2, batch)
        m3 = global_mean_pool(x3, batch)
        m4 = global_mean_pool(x4, batch)
        m5 = global_mean_pool(x5, batch)

        stacked = torch.stack([self.l1(m1), self.l2(m2), self.l3(m3), self.l4(m4), self.l5(m5)], dim=0)
        x = torch.sum(stacked, dim=0)
        return torch.sigmoid(x)


def get_model(model):
    if model == "gcn":
        return NetGCN
    if model == "gsage":
        return NetGraphSage
    if model == "gin":
        return NetGIN
    raise NotImplementedError


def evaluate(model, dataset, steps, dataset_loc, dim):
    ds = TUDataset(root=dataset_loc, name=dataset)
    loader = DataLoader(ds, batch_size=1)

    model = get_model(model)(num_features=ds.num_node_features, dim=dim)

    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    times = []
    for _ in range(steps):
        model.train()

        tm = 0
        for data in loader:
            t = time.perf_counter()
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)

            output = model(data.x, data.edge_index, data.batch)
            loss = F.binary_cross_entropy(output[0][0], data.y[0].float())

            loss.backward()
            optimizer.step()

            tm += time.perf_counter() - t
        times.append(tm)
    return times
