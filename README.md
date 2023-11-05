<p align="center">
<img src="https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/docs/_static/readme_logo.svg" alt="PyNeuraLogic" title="PyNeuraLogic"/>
</p>

[![PyPI version](https://badge.fury.io/py/neuralogic.svg)](https://badge.fury.io/py/neuralogic)
[![License](https://img.shields.io/pypi/l/neuralogic)](https://badge.fury.io/py/neuralogic)
[![Tests Status](https://github.com/LukasZahradnik/PyNeuraLogic/actions/workflows/tests.yml/badge.svg)](https://github.com/LukasZahradnik/PyNeuraLogic/actions/workflows/tests.yml)
[![Code Quality Status](https://github.com/LukasZahradnik/PyNeuraLogic/actions/workflows/black-mypy-flake.yml/badge.svg)](https://github.com/LukasZahradnik/PyNeuraLogic/actions/workflows/black-mypy-flake.yml)
[![Documentation Status](https://readthedocs.org/projects/pyneuralogic/badge/?version=latest)](https://pyneuralogic.readthedocs.io/en/latest/?badge=latest)
[![Tweet](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2FLukasZahradnik%2FPyNeuraLogic)](https://twitter.com/intent/tweet?text=Check%20out:&url=https%3A%2F%2Fgithub.com%2FLukasZahradnik%2FPyNeuraLogic)


[Documentation](https://pyneuralogic.readthedocs.io/en/latest/) | [Examples](#-examples) | [Papers](#-papers)

PyNeuraLogic lets you use Python to write **Differentiable Logic Programs**

---

## About

Logic programming is a declarative coding paradigm in which you declare your logical _variables_ and _relations_ between them. These can be further composed into so-called _rules_ that drive the computation. Such a rule set then forms a _logic program_, and its execution is equivalent to performing logic inference with the rules.

PyNeuralogic, through its [**NeuraLogic**](https://github.com/GustikS/NeuraLogic) backend, then makes this inference process _differentiable_ which, in turn, makes it equivalent to forward propagation in deep learning. This lets you learn numeric parameters that can be associated with the rules, just like you learn weights in neural networks.

<p align="center">
    <a href="https://pyneuralogic.readthedocs.io/en/latest/advanced/database_deep_learning.html">
        <img src="https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/docs/_static/sql_banner.svg" alt="SQL tutorial" title="SQL tutorial"/>
    </a>
</p>


### What is this good for?

Many things! For instance - ever heard of [Graph Neural Networks](https://distill.pub/2021/gnn-intro/) (GNNs)? Well, a _graph_ happens to be a special case of a logical relation - a binary one to be more exact. Now, at the heart of any GNN model there is a so-called _propagation rule_ for passing 'messages' between the neighboring nodes. Particularly, the representation ('message') of a node `X` is calculated by aggregating the previous representations of adjacent nodes `Y`, i.e. those with an `edge` between `X` and `Y`.

Or, a bit more 'formally':

```logtalk
Relation.message2(Var.X) <= (Relation.message1(Var.Y), Relation.edge(Var.Y, Var.X))
```

...and that's the actual _code_! Now for a classic learnable GNN layer, you'll want to add some weights, such as

```logtalk
Relation.message2(Var.X)[5,10] <= (Relation.message1(Var.Y)[10,20], Relation.edge(Var.Y, Var.X))
```

to project your `[20,1]` input node embeddings ('message1') through a learnable ``[10,20]`` layer before the aggregation, and subsequently a `[5,10]` layer after the aggregation.

If you don't like the default settings, you can of course [specify](https://pyneuralogic.readthedocs.io/en/latest/language.html) various additional details, such as the particular aggregation and activation functions

```logtalk
(R.message2(V.X)[5,10] <= (R.message1(V.Y)[10,20], R.edge(V.Y, V.X))) | [Transformation.RELU, Aggregation.AVG]
```

to instantiate the classic GCN layer specification, which you can directly train now!


### How is it different from other GNN frameworks?

Naturally, PyNeuralogic is by no means limited to GNN models, as the expressiveness of _relational_ logic goes much further beyond graphs. Hence, nothing stops you from playing directly with:
- multiple relations and object types
- hypergraphs, nested graphs, relational databases
- relational pattern matching, various subgraph GNNs
- alternative propagation schemes
- inclusion of logical background knowledge
- and more...

In [PyNeuraLogic](https://dspace.cvut.cz/bitstream/handle/10467/97065/F3-DP-2021-Zahradnik-Lukas-Extending-Graph-Neural-Networks-with-Relational-Logic.pdf?sequence=-1&isAllowed=y), all these ideas take the same form of simple small logic programs. These are commonly highly transparent and easy to understand, thanks to their declarative nature. Consequently, there is no need to design a zoo of blackbox class names for each small modification of the GNN rule - you code directly at the level of the logical principles here!

The [backend engine](https://jair.org/index.php/jair/article/view/11203) then creates the underlying differentiable computation (inference) graphs in a fully automated and dynamic fashion, hence you don't have to care about aligning everything into some static (tensor) operations.


### How does it perform?

While PyNeuraLogic allows you to easily declare highly expressive models with capabilities far [beyond the common GNNs](https://arxiv.org/abs/2007.06286), it does not come at the cost of performance for the basic GNNs either. On the contrary, for a range of common GNN models and applications, such as learning with molecules, PyNeuraLogic is actually _considerably_ faster than the popular GNN frameworks, as demonstrated in our [benchmarks](https://pyneuralogic.readthedocs.io/en/latest/benchmarks.html).

<p align="center">
<img src="https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/docs/_static/benchmark.svg" alt="Benchmark of PyNeuraLogic" title="Benchmark of PyNeuraLogic"/>
</p>

<br>

We hope you'll find the framework useful in designing _your own_ deep **relational** learning ideas beyond the GNNs!
Please let us know if you need some guidance or would like to cooperate!


## 💡 Getting started


### Installation

To install PyNeuraLogic's latest release from the PyPI repository, use the following command:

```commandline
$ pip install neuralogic
```


### Prerequisites

To use PyNeuraLogic, you need to install the following prerequisites:

```
Python >= 3.8
Java >= 1.8
```

In case you want to use visualization provided in the library, it is required to have [Graphviz](https://graphviz.org/download/) installed.

## 🔬 Examples
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/SimpleXOR.ipynb) [Simple XOR example](https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/SimpleXOR.ipynb)
<br />
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/MolecularGNN.ipynb) [Molecular GNNs](https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/MolecularGNN.ipynb)
<br />
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/RecursiveXORGeneralization.ipynb) [Recursive XOR generalization](https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/RecursiveXORGeneralization.ipynb)
<br />
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/Visualization.ipynb) [Visualization](https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/Visualization.ipynb)
<br />
<br />
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/PatternMatching.ipynb) [Subgraph Patterns](https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/PatternMatching.ipynb)
<br />
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/DistinguishingKRegularGraphs.ipynb) [Distinguishing k-regular graphs](https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/DistinguishingKRegularGraphs.ipynb)
<br />
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/DistinguishingNonRegularGraphs.ipynb) [Distinguishing non-regular graphs](https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/DistinguishingNonRegularGraphs.ipynb)

<br />


## 📦 Predefined Modules

PyNeuraLogic has a set of predefined modules to get you quickly started with your experimenting!
It contains, for example, predefined modules for:

- Graph Neural Networks (GNNConv, SAGEConv, GINConv, RGCNConv, ...)
- Meta graphs and meta paths (MetaConv, MAGNN, ...)
- Transformer, LSTM, GRU, RNN, [...and more!](https://pyneuralogic.readthedocs.io/en/latest/zoo.html)

## 📝 Papers

- [Beyond Graph Neural Networks with Lifted Relational Neural Networks](https://arxiv.org/abs/2007.06286) Machine Learning Journal, 2021
- [Lifted Relational Neural Networks](https://arxiv.org/abs/1508.05128) Journal of Artificial Intelligence Research, 2018
- [Lossless compression of structured convolutional models via lifting](https://arxiv.org/abs/2007.06567) ICLR, 2021

## 📘 Articles

- [What is Relational Machine Learning?](https://medium.com/towards-data-science/what-is-relational-machine-learning-afbe4a9c4231)
- [What is Neural-Symbolic Integration?](https://medium.com/towards-data-science/what-is-neural-symbolic-integration-d5c6267dfdb0)
- [From Graph ML to Deep Relational Learning](https://medium.com/towards-data-science/from-graph-ml-to-deep-relational-learning-f07a0dddda89)
- [Beyond Graph Neural Networks with PyNeuraLogic](https://medium.com/towards-data-science/beyond-graph-neural-networks-with-pyneuralogic-c1e6502c46f7)
- [Towards Deep Learning for Relational Databases](https://medium.com/towards-data-science/towards-deep-learning-for-relational-databases-de9adce5bb00)
- [Beyond Transformers with PyNeuraLogic](https://medium.com/towards-data-science/beyond-transformers-with-pyneuralogic-10b70cdc5e45)


## 🎥 Videos

- [Beyond Graph Neural Networks with Lifted Relational Neural Networks
](https://www.youtube.com/watch?v=qA0tQ8jwrlA)
