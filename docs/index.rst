PyNeuraLogic
============


.. toctree::
    :maxdepth: 1
    :hidden:

    installation
    quick_start
    language
    custom_model
    rules_mapping
    evaluation
    zoo
    advanced
    examples
    benchmarks

.. toctree::
    :maxdepth: 1
    :caption: Beyond GNNs
    :hidden:

    beyond/hypergraphs
    beyond/heterophily


.. toctree::
    :caption: Package Reference
    :hidden:

    package/neuralogic
    package/core/neuralogic.core
    package/neuralogic.dataset
    package/neuralogic.db
    package/neuralogic.inference
    package/neuralogic.logging
    package/nn/neuralogic.nn
    package/optim/neuralogic.optim
    package/utils/neuralogic.utils



.. image:: https://badge.fury.io/py/neuralogic.svg
    :target: https://badge.fury.io/py/neuralogic

.. image:: https://img.shields.io/pypi/l/neuralogic
    :target: https://badge.fury.io/py/neuralogic

.. image:: https://github.com/LukasZahradnik/PyNeuraLogic/actions/workflows/tests.yml/badge.svg
    :target: https://github.com/LukasZahradnik/PyNeuraLogic/actions/workflows/tests.yml

.. image:: https://readthedocs.org/projects/pyneuralogic/badge/?version=latest
    :target: https://pyneuralogic.readthedocs.io/en/latest/?badge=latest


PyNeuraLogic lets you use Python to create Differentiable Logic Programs

--------

Logic programming is a declarative coding paradigm in which you declare your logical *variables* and *relations* between them. These can be further composed into so-called *rules* that drive the computation. Such a rule set then forms a *logic program*, and its execution is equivalent to performing logic inference with the rules.

PyNeuralogic, through its `NeuraLogic <https://github.com/GustikS/NeuraLogic>`_ backend, then makes this inference process
*differentiable* which, in turn, makes it equivalent to forward propagation in deep learning. This lets you learn numeric parameters that can be associated with the rules, just like you learn weights in neural networks.

What is this good for?
**********************

Many things! For instance - ever heard of `Graph Neural Networks <https://distill.pub/2021/gnn-intro/>`_ (GNNs)? Well, a *graph*
happens to be a special case of a logical relation - a binary one to be more exact. Now, at the heart of any GNN model there is
a so-called *propagation rule* for passing 'messages' between the neighboring nodes. Particularly, the representation ('message')
of a node :code:`X` is calculated by aggregating the representations of adjacent nodes :code:`Y`, i.e. those with an :code:`edge`
between :code:`X` and :code:`Y`.

Or, a bit more 'formally':

.. code-block::

    Relation.node2(Var.X) <= (Relation.node1(Var.Y), Relation.edge(Var.Y, Var.X))

...and that's the actual *code*! Now for a classic learnable GNN layer, you'll want to add some numeric parameters, such as

.. code-block::

    Relation.node2(Var.X)[5,10] <= (Relation.node1(Var.Y)[10,20], Relation.edge(Var.Y, Var.X))

to project your :code:`[1,20]` input node embeddings through a learnable :code:`[10,20]` layer before the aggregation,
and subsequently a :code:`[5,10]` layer after the aggregation. The particular aggregation and transformation functions, as
well as other details, can naturally be `specified further <https://pyneuralogic.readthedocs.io/en/latest/language.html>`_,
but you can as well leave it default like we did here with your first, fully functional GNN layer!

How is it different from other GNN frameworks?
**********************************************

Naturally, PyNeuralogic is by no means limited to GNN models, as the expressiveness of *relational* logic goes much further beyond graphs. So nothing stops you from playing directly with:

- multiple relations and object types
- hypergraphs, nested graphs, relational databases
- alternative propagation schemes
- direct sub-structure (pattern) matching
- inclusion of logical background knowledge
- and more...

In `PyNeuraLogic <https://dspace.cvut.cz/bitstream/handle/10467/97065/F3-DP-2021-Zahradnik-Lukas-Extending-Graph-Neural-Networks-with-Relational-Logic.pdf?sequence=-1&isAllowed=y>`_,
all these ideas take the same form of simple small logic programs. These are commonly highly transparent and easy to understand, thanks to their declarative nature. Consequently,
there is no need to design a new blackbox class name for each small modification of the GNN rule - you code directly at the level of the logical principles here!

The backend engine then creates the underlying differentiable computation (inference) graphs in a fully automated and dynamic fashion, hence you don't have to care about aligning everything into some (static) tensor operations.
This gives you considerably more expressiveness, and, perhaps surprisingly, sometimes even `performance <https://arxiv.org/abs/2007.06286>`_.


We hope you'll find the framework useful in designing your own deep **relational** learning ideas beyond the GNNs!
Please let us know if you need some guidance or would like to cooperate!


Examples
********

.. |mutagcolab| image:: https://colab.research.google.com/assets/colab-badge.svg
                :alt: Open in Colab
                :target: https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/MolecularGNN.ipynb

.. |xorcolab| image:: https://colab.research.google.com/assets/colab-badge.svg
                :alt: Open in Colab
                :target: https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/SimpleXOR.ipynb

.. |recxorcolab| image:: https://colab.research.google.com/assets/colab-badge.svg
                :alt: Open in Colab
                :target: https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/RecursiveXORGeneralization.ipynb

.. |visual| image:: https://colab.research.google.com/assets/colab-badge.svg
                :alt: Open in Colab
                :target: https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/Visualization.ipynb

.. |patterncolab| image:: https://colab.research.google.com/assets/colab-badge.svg
                :alt: Open in Colab
                :target: https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/PatternMatching.ipynb

.. |kregularcolab| image:: https://colab.research.google.com/assets/colab-badge.svg
                :alt: Open in Colab
                :target: https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/DistinguishingKRegularGraphs.ipynb

.. |nonregularcolab| image:: https://colab.research.google.com/assets/colab-badge.svg
                :alt: Open in Colab
                :target: https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/DistinguishingNonRegularGraphs.ipynb


- |mutagcolab| `Molecular GNNs <https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/MolecularGNN.ipynb>`_
- |xorcolab| `Simple XOR example <https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/SimpleXOR.ipynb>`_
- |recxorcolab| `Recursive XOR Generalization <https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/RecursiveXORGeneralization.ipynb>`_
- |visual| `Visualization <https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/Visualization.ipynb>`_

- |patterncolab| `Pattern Matching <https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/PatternMatching.ipynb>`_
- |kregularcolab| `Distinguishing k-regular graphs <https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/DistinguishingKRegularGraphs.ipynb>`_
- |nonregularcolab| `Distinguishing non-regular graphs <https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/DistinguishingNonRegularGraphs.ipynb>`_


Papers
******

- `Beyond Graph Neural Networks with Lifted Relational Neural Networks <https://arxiv.org/abs/2007.06286>`_ Machine Learning Journal, 2021
- `Lifted Relational Neural Networks <https://arxiv.org/abs/1508.05128>`_ Journal of Artificial Intelligence Research, 2018
- `Lossless compression of structured convolutional models via lifting <https://arxiv.org/abs/2007.06567>`_ ICLR, 2021
