import json
import os

import numpy as np


import matplotlib.pyplot as plt


if __name__ == "__main__":
    frameworks = {
        "pyneuralogic": "**PyNeuraLogic**",
        "pyg": "PyTorch Geometric",
        "dgl": "Deep Graph Library",
        "spektral": "Spektral",
    }

    dataset = "MUTAG"
    models = ["gcn", "gsage", "gin"]
    base_dir = os.path.dirname(os.path.abspath(__file__))

    plt.rcParams["axes.titlepad"] = 20

    time_data = [[] for _ in frameworks]
    build_time = []

    for i, framework in enumerate(frameworks):
        for model in models:
            path = os.path.join(base_dir, "results", dataset, model, f"{framework}.json")

            with open(path) as fp:
                framework_data = json.load(fp)

                if framework == "pyneuralogic":
                    build_time.append(framework_data["build_time"])
                time_data[i].append(sum(framework_data["times"]) / framework_data["steps"])

    x = np.arange(len(models))
    width = 0.2
    colors = ["#fb86ad", "#5595e4", "#87dcd7", "#c7a0f8"]

    plt.ylabel("time (s)")
    # plt.title(f"Average time per epoch ({dataset} dataset)")

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

    plt.legend([fw.replace("*", "") for fw in frameworks.values()])
    plt.savefig(f"{dataset}.svg")

    plt.clf()

    fw_name_len = max(len(framework) for framework in frameworks.values()) + 2
    line = lambda char="-": f"+{'+'.join(char * col_len for col_len in [fw_name_len, 12, 12, 12])}+"
    fill = lambda text, max_len=12: f"{text}{' ' * (max_len - len(text))}"

    data_line = lambda fw, data: f"|{fill(fw, fw_name_len)}|{'|'.join(data)}|"

    #
    print("""Average Time Per Epoch\n----------------------\n""")

    print(line())
    print(data_line(" ", [fill(fw) for fw in ["GCN", "GraphSAGE", "GIN"]]))
    print(line("="))

    for i, fw in enumerate(reversed(frameworks.values())):
        if fw.startswith("**"):
            print(data_line(fw, [fill(f"**{d:.4f}s**") for d in time_data[len(time_data) - 1 - i]]))
        else:
            print(data_line(fw, [fill(f"{d:.4f}s") for d in time_data[len(time_data) - 1 - i]]))
        print(line())

    print("\nGraph Build Time\n----------------\n")

    print(line())
    print(data_line(" ", [fill(fw) for fw in ["GCN", "GraphSAGE", "GIN"]]))
    print(line("="))
    print(data_line("PyNeuraLogic", [fill(f"{d:.4f}s") for d in build_time]))
    print(line())
