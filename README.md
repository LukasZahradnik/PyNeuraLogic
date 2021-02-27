# PyNeuraLogic

[![PyPI version](https://badge.fury.io/py/neuralogic.svg)](https://badge.fury.io/py/neuralogic)
[![License](https://img.shields.io/pypi/l/neuralogic)](https://badge.fury.io/py/neuralogic)



[Documentation](https://pyneuralogic.readthedocs.io/en/latest/) | [Examples](#examples) | [Papers](https://github.com/GustikS/NeuraLogic#papers)

PyNeuraLogic is a framework built on top of [NeuraLogic](https://github.com/GustikS/NeuraLogic) which combines relational and deep learning.

---

### General



##### Supported backends (WIP):
- [PyTorch Geometric (PyG)](https://github.com/rusty1s/pytorch_geometric)
- [DyNet](https://github.com/clab/dynet)
- [Deep Graph Library (DGL)](https://github.com/dmlc/dgl)

### Getting started

### Prerequisites

To use PyNeuraLogic, you need to have installed the following prerequisites.

```
Python >= 3.7
Java 1.8
```

### Installation

To install PyNeuraLogic's latest release from the PyPI repository, use the following command.

```commandline
$ pip install neuralogic
```

### How to use

None of the following backends are included in PyNeuraLogic's installation. You have to install the ones that you are planning to utilize manually.

#### With PyTorch Geometric

```python

```

#### With DyNet

```python
import dynet as dy
from neuralogic import data
from neuralogic.dynet import NeuraLogicLayer

dataset = data.XOR  # Use one of the default datasets in the project in the/datasets/ folder

layer = NeuraLogicLayer(dataset.weights)  # Create an instance of NeuraLogicLayer with weights from the dataset
trainer = dy.AdamTrainer(layer.model, alpha=0.001)

for sample in dataset.samples:  # Learn on each sample
    dy.renew_cg(immediate_compute=False, check_validity=False)
    label = dy.scalarInput(sample.target)
    
    graph_output = layer.build_sample(sample)
    
    loss = dy.squared_distance(graph_output, label)
    loss.forward()
    loss.backward()
    trainer.update()
```

#### With Deep Graph Library

```python

```

### Development

```commandline

```
