import argparse
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-fw", help="framework", type=str)
    parser.add_argument("-model", help="type of model (gcn, gin, gsage)", type=str)
    parser.add_argument("-out", nargs="?", help="path to output folder", type=str)
    parser.add_argument("-ts", nargs="?", help="number of training steps", type=int)
    parser.add_argument("-ds", nargs="?", help="dataset", type=str)

    args = parser.parse_args()

    framework = args.fw
    steps = args.ts or 300
    dataset = args.ds or "MUTAG"
    out = args.out or os.path.join(os.path.abspath(os.path.dirname(__file__)), "results", dataset, args.model)
    dataset_loc = os.path.join(os.path.abspath(os.path.dirname(__file__)), "datasets")

    if not os.path.exists(out):
        os.makedirs(out)

    if framework == "pyg":
        from pyg_benchmark import evaluate
    if framework == "dgl":
        from dgl_benchmark import evaluate
    if framework == "pyneuralogic":
        from pyneuralogic_benchmark import evaluate
    if framework == "spektral":
        from spektral_benchmark import evaluate

    times = evaluate(args.model, dataset, steps, dataset_loc, 10)

    with open(os.path.join(out, f"{framework}.json"), "w") as fp:
        json.dump({"times": times, "steps": steps}, fp)
