import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GCNConv, global_mean_pool, GINConv, SAGEConv

from utils import Results, to_json, export_fold, ResultList, Crossval

torch.set_default_tensor_type("torch.DoubleTensor")
device = torch.device("cpu")


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
        self.conv1 = SAGEConv(num_features, dim, normalize=False, concat=True, bias=False)
        self.conv2 = SAGEConv(dim, dim, normalize=False, concat=True, bias=False)

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


class Evaluator:
    @staticmethod
    def train(model, loader, optimizer):
        model.train()

        loss_all = 0
        outputs = []
        labels = []

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(data.x, data.edge_index, data.batch)

            loss = F.binary_cross_entropy(output[0][0], data.y[0].double())

            if len(data.y) == 1:
                outputs.append(output)
                labels.append(data.y)
            else:
                outputs.extend(output)
                labels.extend(data.y)

            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return Results(outputs, labels)

    @staticmethod
    def test(model, loader):
        model.eval()

        outputs = []
        labels = []

        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            pred = output.max(dim=1)[1]

            if len(data.y) == 1:
                outputs.append(output)
                labels.append(data.y)
            else:
                outputs.extend(output)
                labels.extend(data.y)
        return Results(outputs, labels)

    @staticmethod
    def learn(model, train_dataset, val_dataset, test_dataset, steps=1000, lr=0.000015):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_results = Results()
        best_val_results.loss = 1e10
        best_train_results = None
        best_test_results = None

        cumtime = 0

        for epoch in range(steps):
            start = time.time()
            train_results = Evaluator.train(model, train_dataset, optimizer)

            val_results = Evaluator.test(model, val_dataset)
            test_results = Evaluator.test(model, test_dataset)

            end = time.time()

            if val_results.loss < best_val_results.loss:
                print(f"improving validation loss to {val_results.loss} at epoch {epoch}")
                best_val_results = val_results
                best_test_results = test_results
                best_train_results = train_results
                print(f"storing respective test results with accuracy {best_test_results.accuracy}")

            elapsed = end - start
            cumtime += elapsed

            print(
                "Epoch: {:03d}, Train Loss: {:.7f}, "
                "Train Acc: {:.7f}, Val Acc: {:.7f}, Test Acc: {:.7f}".format(
                    epoch, train_results.loss, train_results.accuracy, val_results.accuracy, test_results.accuracy
                )
                + " elapsed: "
                + str(elapsed)
            )
        return best_train_results, best_val_results, best_test_results, cumtime / steps

    @staticmethod
    def crossvalidate(
        model_string,
        folds,
        outpath: Optional[Path],
        num_node_features: int,
        steps: int = 1000,
        lr: float = 0.000015,
        dim: int = 10,
    ):
        if outpath is None:
            outpath = Path("./out")

        train_results = []
        val_results = []
        test_results = []
        times = []

        for train_fold, val_fold, test_fold in folds:
            model = Evaluator.get_model(model_string, dim, num_node_features)

            best_train_results, best_val_results, best_test_results, elapsed = Evaluator.learn(
                model, train_fold, val_fold, test_fold, steps, lr
            )

            train_results.append(best_train_results)
            val_results.append(best_val_results)
            test_results.append(best_test_results)
            times.append(elapsed)

            train = to_json(ResultList(train_results, times=times))
            export_fold(train, outpath / "train")

            test = to_json(ResultList(test_results))
            export_fold(test, outpath / "test")

        return Crossval(train_results, val_results, test_results, times)

    @staticmethod
    def get_model(string: str, dim: int, num_node_features: int):
        if string == "gcn":
            model = NetGCN(num_node_features, dim=dim).to(device)
        elif string == "gin":
            model = NetGIN(num_node_features, dim=dim).to(device)
        elif string == "gsage":
            model = NetGraphSage(num_node_features, dim=dim).to(device)
        else:
            raise NotImplementedError("Unknown model")
        return model
