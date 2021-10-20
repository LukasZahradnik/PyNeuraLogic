# PyNeuraLogic

[![PyPI version](https://badge.fury.io/py/neuralogic.svg)](https://badge.fury.io/py/neuralogic)
[![License](https://img.shields.io/pypi/l/neuralogic)](https://badge.fury.io/py/neuralogic)
[![Documentation Status](https://readthedocs.org/projects/pyneuralogic/badge/?version=latest)](https://pyneuralogic.readthedocs.io/en/latest/?badge=latest)



[Documentation](https://pyneuralogic.readthedocs.io/en/latest/) | [Examples](#Examples) | [Papers](https://github.com/GustikS/NeuraLogic#papers)

PyNeuraLogic lets you use Python to create **Differentiable Logic Programs**


[comment]: <> (PyNeuraLogic is a framework built on top of [NeuraLogic]&#40;https://github.com/GustikS/NeuraLogic&#41; which combines relational and deep learning.)

---

## About

Logic programming is a declarative coding paradigm in which you declare your logical _variables_ and _relations_ between them. A relation applied to a tuple of variables is then called an _atom_. These can be further composed into so-called _rules_ that drive the computation. Such rule-set is called a _logic program_, and its execution is equivalent to logic inference over the rules.

PyNeuralogic, with its [NeuraLogic](https://github.com/GustikS/NeuraLogic) backend, then makes this inference process differentiable, which lets you learn numeric parameters that can be associated with it.

### What is this good for?

Many things! For instance, ever heard of [Graph Neural Networks](https://distill.pub/2021/gnn-intro/) (GNNs)? Well, a graph happens to be a special case of a logical relation - a binary one to be exact. Now, at the heart of any GNN model there is a _propagation rule_ for passing 'messages' between the adjacent nodes. Particularly, the representation ('message') of a node `X` is calculated by aggregating the representations of nodes `Y`, such that there is an edge between `X` and `Y`. 

Or, a bit more formally:

``` Atom.node2(Var.X) <= Atom.node1(Var.Y), Atom.edge(Var.X,Var.Y)```

...and that's the actual code! Now for a classic learnable GNN layer, you'll want to add some parameters, such as

``` Atom.node2(Var.X)[5,10] <= Atom.node1(Var.Y)[10,20], Atom.edge(Var.X,Var.Y)```

to project your node embeddings through a learnable ``[10,20]`` layer before the aggregation, and also `[5,10]` after the aggregation. The particular aggregation and activation functions, as well as other details, can naturally be [specified further](https://pyneuralogic.readthedocs.io/en/latest/language.html), but you can as well leave it default like we did here with your first, fully functional GNN layer!

### What is the difference from other GNN frameworks?

Naturally, PyNeuralogic is by no means limited to GNN models, as the expressiveness of relational logic goes much further than graphs. So nothing stops you from playing directly with:
- multiple relations and node types
- hypergraphs, nested graphs, relational databases
- alternative propagation schemes
- direct sub-structure (pattern) matching
- inclusion of logical background knowledge
- and more...

In PyNeuraLogic, all these concepts take the same form of simple small logic programs. These are commonly highly transparent and easy to understand, thanks to their declarative nature. Consequently, there is no need to design a new blackbox class name for each new small modification of the GNN rule, since you code directly at the level of the logical principles here.

The underlying engine then creates differentiable computation graphs in a fully automated and dynamic fashion, hence you don't have to care about how to align everything into some (static) tensor operations.
This gives you considerably more expressiveness, and, somewhat surprisingly, sometimes even [performance](https://github.com/GustikS/GNNwLRNNs).


We hope you'll find the framework useful in designing your own deep **relational** learning ideas beyond the current GNNs!

[comment]: <> (PyNeuraLogic allows users to encode machine learning problems via parameterized, rule-based constructs. Said constructs are based on a custom declarative language that follows a logic programming paradigm.)

## Getting started

[comment]: <> (### Supported backends)

[comment]: <> (PyNeuraLogic currently supports following backends &#40;to some extent&#41;, which have to be installed separately:)

[comment]: <> (- [DyNet]&#40;https://github.com/clab/dynet&#41;)

[comment]: <> (- Java)

[comment]: <> (- [PyTorch Geometric]&#40;https://github.com/rusty1s/pytorch_geometric&#41;)

### Prerequisites

To use PyNeuraLogic, you need to install the following prerequisites:

```
Python >= 3.7
Java 1.8
```

### Installation

To install PyNeuraLogic's latest release from the PyPI repository, use the following command:

```commandline
$ pip install neuralogic
```

## Examples

- [XOR Example](https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/IntroductionIntoPyNeuraLogic.ipynb)
- [Pattern Matching](https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/PatternMatching.ipynb)
- [Distinguishing k-regular graphs](https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/DistinguishingKRegularGraphs.ipynb)
- [Distinguishing non-regular graphs](https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/DistinguishingNonRegularGraphs.ipynb)
