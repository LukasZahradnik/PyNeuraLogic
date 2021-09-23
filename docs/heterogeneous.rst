Heterogeneous Graphs
====================


Most GNN models do not consider graphs being heterogeneous. Via PyNeuraLogic, we can easily encode heterogeneous
graphs with an arbitrary number of node and edge classes.


.. image:: https://raw.githubusercontent.com/LukasZahradnik/PyNeuraLogic/master/docs/images/heterograph.png
    :width: 500
    :alt: Heterogeneous graph
    :align: center


.. code-block:: Python

    Atom.type(1, Term.RED),
    Atom.type(2, Term.RED),
    Atom.type(3, Term.BLUE),
    Atom.type(4, Term.BLUE),



.. code-block:: Python

    Atom.h(Var.X) <= (
        Atom.feature(Var.Y),
        Atom.type(Var.X, Var.Type),
        Atom.type(Var.Y, Var.Type),
        Atom.edge(Var.X, Var.Y),
    )


.. code-block:: Python

    Atom.type_feature(Term.RED)[[1, 2, 3]]

.. code-block:: Python

    Atom.h(Var.X) <= (
        Atom.feature(Var.Y),
        Atom.type_feature(Var.Y),
        Atom.type(Var.Y, Var.Type),
        Atom.edge(Var.X, Var.Y),
    )
