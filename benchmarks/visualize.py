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

    dataset = "MUTAG"
    models = ["gcn", "gsage", "gin"]
    base_dir = os.path.dirname(os.path.abspath(__file__))

    time_data = [[] for _ in frameworks]

    for i, framework in enumerate(frameworks):
        for model in models:
            path = os.path.join(base_dir, "results", dataset, model, f"{framework}.json")

            with open(path) as fp:
                framework_data = json.load(fp)
                time_data[i].append(sum(framework_data["times"]) / framework_data["steps"])
        print(framework, time_data[i])

    x = np.arange(len(models))
    colors = ["#fb86ad", "#5595e4", "#87dcd7", "#c7a0f8"]

    for i, fw_data in enumerate(time_data):
        plt.bar(x + i * 0.2, fw_data, width=0.2, edgecolor=colors[i], color=colors[i])
    plt.savefig("mutag.svg")

    plt.clf()
