import json
import os

import numpy as np


import matplotlib.pyplot as plt


if __name__ == "__main__":
    frameworks = {
        "pyneuralogic": "PyNeuraLogic",
        "pyg": "PyG",
        "dgl": "DGL",
        "spektral": "Spektral",
    }

    dataset = "ENZYMES"
    models = ["gcn", "gsage", "gin"]
    base_dir = os.path.dirname(os.path.abspath(__file__))

    plt.rcParams["axes.titlepad"] = 20

    time_data = [[] for _ in frameworks]

    for i, framework in enumerate(frameworks):
        for model in models:
            path = os.path.join(base_dir, "results", dataset, model, f"{framework}.json")

            with open(path) as fp:
                framework_data = json.load(fp)
                time_data[i].append(sum(framework_data["times"]) / framework_data["steps"])
        print(framework, time_data[i])

    x = np.arange(len(models))
    width = 0.2
    colors = ["#fb86ad", "#5595e4", "#87dcd7", "#c7a0f8"]

    plt.ylabel("time (s)")
    plt.title(f"Average time per epoch ({dataset} dataset)")

    x_ticks = []
    for tick in x:
        x_ticks.append(tick - width / 2)
        x_ticks.append(tick + width * len(frameworks) / 2 - width / 2)
        x_ticks.append(tick + width * len(frameworks) - width / 2)

    x_labels = ["", "GCN", "", "", "GraphSAGE", "", "", "GIN", ""]

    plt.xticks(x_ticks, x_labels)

    ax = plt.gca()
    for i, tick in enumerate(filter(lambda x: x.get_marker() == 3, ax.xaxis.get_ticklines())):
        if x_labels[i] != "":
            tick.set_alpha(0)

    for i, fw_data in enumerate(time_data):
        plt.bar(x + i * 0.2, fw_data, width=width, edgecolor=colors[i], color=colors[i])
    plt.savefig(f"{dataset}.svg")

    plt.clf()
