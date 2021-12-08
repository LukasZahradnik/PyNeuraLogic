import json
import os
import pickle
import statistics
from os import listdir
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

import dgl

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from random import shuffle

from neuralogic.core import Dataset

torch.set_default_tensor_type("torch.DoubleTensor")


device = torch.device("cpu")


class Crossval:
    def __init__(self, train_results, val_results, test_results, times=None):
        self.train_loss_mean, self.train_loss_std = self.aggregate_loss(train_results)
        self.train_acc_mean, self.train_acc_std = self.aggregate_acc(train_results)

        self.val_loss_mean, self.val_loss_std = self.aggregate_loss(val_results)
        self.val_acc_mean, self.val_acc_std = self.aggregate_acc(val_results)

        self.test_loss_mean, self.test_loss_std = self.aggregate_loss(test_results)
        self.test_acc_mean, self.test_acc_std = self.aggregate_acc(test_results)

        if times:
            self.time_per_step = sum(times) / len(times)

    def aggregate_loss(self, results):
        losses = [res.loss for res in results]
        return np.mean(losses), np.std(losses)

    def aggregate_acc(self, results):
        accuracies = [res.accuracy for res in results]
        return statistics.mean(accuracies), statistics.stdev(accuracies)


class Results:
    def __init__(self, pred=None, lab=None, loss_fcn=F.binary_cross_entropy, loss=None):
        self.loss = 0
        self.accuracy = 0

        if loss and lab and pred:
            self.loss = sum(loss) / len(lab)
            self.accuracy = self.acc(pred, lab, True)
        elif pred and lab:
            for p, l in zip(pred, lab):
                if p.ndim == 1:  # batching changes this
                    self.loss += loss_fcn(p, l)
                else:
                    self.loss += loss_fcn(p[0][0], l[0].double())
            self.loss /= len(lab)
            self.loss = float(self.loss)
            self.accuracy = self.acc(pred, lab)

    def acc(self, output, labels, native=False):
        correct = 0
        for out, lab in zip(output, labels):
            if native:
                out_ = out
            elif len(out) == 1:
                out_ = out
            else:
                out_ = out[0][0]

            pred = 1 if out_ > 0.5 else 0

            if native:
                lab_ = lab
            elif lab.ndim == 0:
                lab_ = lab
            else:
                lab_ = lab[0]

            if pred == int(lab_):
                correct += 1
        return correct / len(labels)

    def increment(self, other):
        self.loss = other.loss
        self.accuracy = other.accuracy


class ResultList:
    def __init__(self, results, times=None):
        self.folds = [to_json(res) for res in results]
        if times:
            self.times = times


def load_obj(name):
    with open(name, "rb") as f:
        return pickle.load(f)


def convert_to_dgl(graphs):
    dgl_graphs = []
    for graph in graphs:
        u, v = graph["edge_index"]
        g = dgl.graph((u, v))
        g.ndata["x"] = graph["x"].to(torch.float64)
        g.y = graph.y
        dgl_graphs.append(g)
    return dgl_graphs


def load_dataset_folds_external(path, suffix="_graphs.pkl", batch=1, framework="pyg"):
    folds = []

    for fold in sorted(listdir(path)):
        if fold.startswith("fold"):
            if framework == "pyneuralogic":
                folds.append(
                    [
                        Dataset(examples_file=os.path.join(path, fold, "trainExamples.txt")),
                        Dataset(examples_file=os.path.join(path, fold, "valExamples.txt")),
                        Dataset(examples_file=os.path.join(path, fold, "testExamples.txt")),
                    ]
                )

                continue

            train_fold = load_obj(path + "/" + fold + "/train" + suffix)
            val_fold = load_obj(path + "/" + fold + "/val" + suffix)
            test_fold = load_obj(path + "/" + fold + "/test" + suffix)

            if framework == "dgl":
                train_fold = convert_to_dgl(train_fold)
                val_fold = convert_to_dgl(val_fold)
                test_fold = convert_to_dgl(test_fold)

                folds.append((train_fold, val_fold, test_fold))
            else:
                folds.append(
                    [
                        DataLoader(train_fold, batch_size=batch),
                        DataLoader(val_fold, batch_size=batch),
                        DataLoader(test_fold, batch_size=batch),
                    ]
                )
    return folds


def load_dataset_folds(path, batch=1, folds=10):
    dataset = load_obj(path)

    shuffle(dataset)
    skf = StratifiedKFold(n_splits=folds)
    labels = [lab.y.numpy() for lab in dataset]

    folds = []
    for train_idx, test_idx in skf.split(np.zeros(len(dataset)), labels):
        train_fold_tmp = [dataset[i] for i in train_idx]

        y_train = [lab.y.numpy() for lab in train_fold_tmp]

        train_fold, val_fold, _, _ = train_test_split(
            train_fold_tmp, y_train, stratify=y_train, test_size=0.1, random_state=1
        )

        test_fold = [dataset[i] for i in test_idx]
        folds.append(
            [
                DataLoader(train_fold, batch_size=batch),
                DataLoader(val_fold, batch_size=batch),
                DataLoader(test_fold, batch_size=batch),
            ]
        )
    return folds


def to_json(obj):
    return json.dumps(obj.__dict__, indent=4)


def export_fold(content, outpath: Path):
    path = str(outpath) + ".json"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        f.writelines(content)
        f.close()
