import argparse
import json
import os
from pathlib import Path

from utils import load_dataset_folds_external

if __name__ == "__main__":
    num_classes = 2

    parser = argparse.ArgumentParser()

    parser.add_argument("-fw", help="framework", type=str)
    parser.add_argument("-sd", help="path to dataset for learning", type=str)
    parser.add_argument("-model", help="type of model (gcn,gin)", type=str)
    parser.add_argument("-out", help="path to output folder", type=str)

    parser.add_argument("-xval", nargs="?", help="number fo folds for crossval", type=int)
    parser.add_argument("-lr", nargs="?", help="learning rate for Adam", type=float)
    parser.add_argument("-ts", nargs="?", help="number of training steps", type=int)
    parser.add_argument("-batch", nargs="?", help="size of minibatch", type=int)
    parser.add_argument("-dim", nargs="?", help="dimension of hidden layers", type=int)
    parser.add_argument("-filename", nargs="?", help="filename with example data", type=str)
    parser.add_argument("-limit", nargs="?", help="dummy for compatibility wih lrnns", type=str)  # dummy

    args = parser.parse_args()

    steps = args.ts or 1000
    folds = args.xval or 10

    framework = args.fw

    filename = args.filename or "_graphs.pkl"
    dataset_folds = load_dataset_folds_external(args.sd, suffix=filename, framework=framework)

    num_node_features = 51

    if framework == "pyg":
        from pyg_benchmark import Evaluator
    if framework == "dgl":
        from dgl_benchmark import Evaluator

    cross = Evaluator.crossvalidate(args.model.lower(), dataset_folds, Path(args.out), num_node_features, steps)

    content = json.dumps(cross.__dict__, indent=4)
    outp = args.out

    with open(os.path.join(outp, "crossval.json"), "w") as f:
        f.writelines(content)
