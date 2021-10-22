<p align="center">
<img src="https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/docs/_static/readme_logo.svg" alt="PyNeuraLogic" title="PyNeuraLogic"/>
</p>
  
[![PyPI version](https://badge.fury.io/py/neuralogic.svg)](https://badge.fury.io/py/neuralogic)
[![License](https://img.shields.io/pypi/l/neuralogic)](https://badge.fury.io/py/neuralogic)
[![Documentation Status](https://readthedocs.org/projects/pyneuralogic/badge/?version=latest)](https://pyneuralogic.readthedocs.io/en/latest/?badge=latest)



[Documentation](https://pyneuralogic.readthedocs.io/en/latest/) | [Examples](#Examples) | [Papers](https://github.com/GustikS/NeuraLogic#papers)

PyNeuraLogic lets you use Python to create **Differentiable Logic Programs**


[comment]: <> (PyNeuraLogic is a framework built on top of [NeuraLogic]&#40;https://github.com/GustikS/NeuraLogic&#41; which combines relational and deep learning.)

---

## About

Logic programming is a declarative coding paradigm in which you declare your logical _variables_ and _relations_ between them. These can be further composed into so-called _rules_ that drive the computation. Such a rule set then forms a _logic program_, and its execution is equivalent to performing logic inference with the rules.

PyNeuralogic, through its [NeuraLogic](https://github.com/GustikS/NeuraLogic) backend, then makes this inference process _differentiable_, which lets you learn numeric parameters that can be associated with the rules.

### What is this good for?

Many things! For instance - ever heard of [Graph Neural Networks](https://distill.pub/2021/gnn-intro/) (GNNs)? Well, a _graph_ happens to be a special case of a logical relation - a binary one to be more exact. Now, at the heart of any GNN model there is a so-called _propagation rule_ for passing 'messages' between the neighboring nodes. Particularly, the representation ('message') of a node `X` is calculated by aggregating the previous representations of adjacent nodes `Y`, i.e. those with an `edge` between `X` and `Y`. 

Or, a bit more 'formally':

```
Relation.node2(Var.X) <= (Relation.node1(Var.Y), Relation.edge(Var.X,Var.Y))
```

...and that's the actual _code_! Now for a classic learnable GNN layer, you'll want to add some parameters, such as

```
Relation.node2(Var.X)[5,10] <= (Relation.node1(Var.Y)[10,20], Relation.edge(Var.X,Var.Y))
```

to project your `[1,20]` input node embeddings ('messages') through a learnable ``[10,20]`` layer before the aggregation, and subsequently a `[5,10]` layer after the aggregation. The particular aggregation and activation functions, as well as other details, can naturally be [specified further](https://pyneuralogic.readthedocs.io/en/latest/language.html), but you can as well leave them default like we did here with your first, fully functional GNN layer!

### How is it different from other GNN frameworks?

Naturally, PyNeuralogic is by no means limited to GNN models, as the expressiveness of _relational_ logic goes much further beyond graphs. So nothing stops you from playing directly with:
- multiple relations and object types
- hypergraphs, nested graphs, relational databases
- alternative propagation schemes
- direct sub-structure (pattern) matching
- inclusion of logical background knowledge
- and more...

In [PyNeuraLogic](https://dspace.cvut.cz/bitstream/handle/10467/97065/F3-DP-2021-Zahradnik-Lukas-Extending-Graph-Neural-Networks-with-Relational-Logic.pdf?sequence=-1&isAllowed=y), all these ideas take the same form of simple small logic programs. These are commonly highly transparent and easy to understand, thanks to their declarative nature. Consequently, there is no need to design a new blackbox class name for each small modification of the GNN rule - you code directly at the level of the logical principles here!

The [backend engine](https://jair.org/index.php/jair/article/view/11203) then creates the underlying differentiable computation (inference) graphs in a fully automated and dynamic fashion, hence you don't have to care about aligning everything into some (static) tensor operations.
This gives you considerably more expressiveness, and, perhaps surprisingly, sometimes even [performance](https://arxiv.org/abs/2007.06286).


We hope you'll find the framework useful in designing your own deep **relational** learning ideas beyond the GNNs!
Please let us know if you need some guidance or would like to cooperate!

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
