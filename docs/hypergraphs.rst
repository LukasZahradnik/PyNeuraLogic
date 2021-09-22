Hypergraph Neural Networks
==========================


A hypergraph is a generalization of a simple graph :math:`G = (V, E)`, where :math:`V` is a set of vertices
and :math:`E` is a set of edges (hyperedges) connecting an arbitrary number of vertices.

Representation of hyperedges
############################

When we encode input data (graph) in the form of logic data format (i.e., ground atoms),
we can represent regular edges, for example, as :code:`Atom.edge(1, 2)`.

This form of representation can be simply extended to express hyperedges by adding terms for each connected
vertex by the hyperedge. For example, graph :math:`G = (V, E)`, where :math:`V = \{1, 2, 3, 4, 5\}`
and :math:`E = \{\{1, 2\}, \{3, 4, 5\}, \{1, 2, 3, 4\}\}` can be represented as:

.. code-block::

    Atom.edge(1, 2),
    Atom.edge(3, 4, 5),
    Atom.edge(1, 2, 3, 4),


Propagation on hyperedges
#########################

The propagation through standard edges can be similarly extended to support propagation through hyperedges.


.. code-block::

    Atom.h(Var.X) <= (Atom.feature(Var.Y), Atom.edge(Var.X, Var.Y))


The propagation through standard edges above, where :code:`Atom.feature` might represent vertex features,
and :code:`Atom.edge` represents an edge, might be extended to support hyperedges (for hyperedge connecting three
vertices) as follows:

.. code-block::

    Atom.h(Var.X) <= (
        Atom.feature(Var.Y),
        Atom.feature(Var.Z),
        Atom.edge(Var.X, Var.Y, Var.Z),
    )


