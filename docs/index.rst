PyNeuraLogic
============


.. toctree::
    :maxdepth: 1
    :hidden:

    installation
    quick_start
    language
    heterogeneous

.. toctree::
    :maxdepth: 1
    :caption: Beyond GNNs
    :hidden:

    hypergraphs
    pattern_propagation
    tree_gnns
    heterophily
    tabnet


.. toctree::
    :caption: Package Reference
    :hidden:

    neuralogic.core
    neuralogic.nn
    neuralogic.utils



.. image:: https://badge.fury.io/py/neuralogic.svg
    :target: https://badge.fury.io/py/neuralogic

.. image:: https://img.shields.io/pypi/l/neuralogic
    :target: https://badge.fury.io/py/neuralogic


PyNeuraLogic is a library that combines relational and deep learning and is built on top of
`NeuraLogic <https://github.com/GustikS/NeuraLogic>`_. The library's main aim is to enhance Graph Neural Networks
(GNNs) with more expressiveness and push their boundaries.

--------

PyNeuraLogic allows users to encode various machine learning problems via parametrized, rule-based constructs.
Said constructs are based on a custom declarative language that follows a logic programming paradigm.


Why PyNeuraLogic
################



Supported backends
##################

| PyNeuraLogic currently supports following backends (to some extent), which have to be installed separately:

- `DyNet <https://github.com/clab/dynet>`_
- Java
- `PyTorch Geometric <https://github.com/rusty1s/pytorch_geometric>`_

Prerequisites
#############

| To use PyNeuraLogic, you need to install the following prerequisites:

.. code-block::

    Python >= 3.7
    Java 1.8


Installation
############

| To install PyNeuraLogic's latest release from the PyPI repository, use the following command:

.. code-block::

    pip install neuralogic


Examples
########


- `XOR Example <https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/IntroductionIntoPyNeuraLogic.ipynb>`_
- `Pattern Matching <https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/PatternMatching.ipynb>`_
- `Distinguishing k-regular graphs <https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/DistinguishingKRegularGraphs.ipynb>`_
- `Distinguishing non-regular graphs <https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/DistinguishingNonRegularGraphs.ipynb>`_
