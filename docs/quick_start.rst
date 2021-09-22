Quick Start
===========

The PyNeuraLogic library serves for learning on structured data. For the purpose of the introduction to the library and
its syntax, we will further discuss use cases on graph structures.

Graph Representation
####################

Graphs can describe entities (vertices) and their relations (edges) which can be useful for various tasks. Graphs are
used as inputs for models and are contained in the :code:`Dataset` class.

The :code:`Dataset` class containing information about graphs can be used in different ways depending on the data format. The
next section will showcase how to represent the following graph (triangle) in two formats - tensor and logic.

.. image:: https://raw.githubusercontent.com/LukasZahradnik/PyNeuraLogic/master/docs/images/simple_graph.png
    :width: 500
    :alt: Simple graph
    :align: center


Tensor Representation
*********************

The tensor format is a familiar format used in many other GNN focused frameworks and libraries. The input graph is
represented in a graph connectivity format, i.g., tensor of shape :code:`[2, num_of_edges]`.

.. code-block:: Python

    from neuralogic.utils.data.dataset import Dataset
    from neuralogic.utils.data.dataset import Data


    data = Data(
        edge_index=[[1, 2], [2, 1], [1, 3], [3, 1], [2, 3], [3, 2]]
    )

    dataset = Dataset(data=[data])


In this example, we are encoding the simple graph (triangle) in the tensor format. The structure of the graph is
encoded in :code:`edge_index` property of the Data class instance. Each :code:`Data` class instance holds information about exactly
one graph. The :code:`Dataset` instance then holds a list of data instances and serves as the input.

.. NOTE::

    We omitted a few :code:`Data` class attributes, such as :code:`x` for the nodes' features encoding, :code:`edge_attr` for the edges'
    features encoding, and :code:`y` and :code:`y_mask` for the target labels encoding.


The tensor representation offers less verbose graph representation but it is more limited in its expressiveness than the logic
format introduced in the next section.

Logic Representation
********************

The logic format utilizes constructs based on relational logic to encode input data - graphs. The input data are represented in the form of ground atoms (facts),
which can be expressed as :code:`Atom.predicate_name(terms)[value]`.

.. code-block:: Python

    from neuralogic.utils.data.dataset import Dataset
    from neuralogic.core import Atom


    dataset = Dataset()

    dataset.add_examples([
        Atom.edge(1, 2), Atom.edge(2, 1), Atom.edge(1, 3),
        Atom.edge(3, 1), Atom.edge(2, 3), Atom.edge(3, 2)
    ])

In this example, we represent the same simple graph (triangle) but in the logic format.

.. NOTE::
    We used the edge as the predicate name (:code:`Atom.edge`) to represent the graph edges. This naming is arbitrary -
    edges and any other input data can have any predicate name. In this documentation, we will stick to *edge* predicate name for
    representing edges and *feature* predicate name for representing features.
