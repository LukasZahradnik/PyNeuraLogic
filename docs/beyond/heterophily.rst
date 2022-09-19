Heterophily Settings
====================

Regular GNN models usually consider homophily in the graph - frequently,
nodes of similar classes are connected with each other. This setting does not capture multiple problems adequately,
where there is a heterophily amongst connected nodes - mainly nodes of different classes are connected, resulting in
low accuracies of classifications.

There have been proposed new methods and models to properly capture problems of such settings, such as
*CPGNN* (`"Graph Neural Networks with Heterophily" <https://arxiv.org/abs/2009.13566>`_)
or *H2GCN* (`"Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs" <https://arxiv.org/abs/2006.11468>`_).


We take into consideration the latter one - the *H2GCN* model, which is specifically built to deal with heterophily
graphs and implement three key design concepts. All of those concepts can be easily represented in PyNeuraLogic
with a few rules. As in other cases, the rule representation can be further manipulated and tweaked without the need of
reimplementing the whole model or digging into an already implemented black box.

1. The Central Node Embedding Separation
########################################

The first key design is separating the embedding of the central node from the embedding of neighbor nodes. This behavior
can be achieved just with two rules, that can be written in the following form:

.. code-block:: Python

    Relation.layer_1(Var.X) <= (Relation.layer_0(Var.Y), Relation.edge(Var.Y, Var.X)),
    Relation.layer_1(Var.X) <= Relation.layer_0(Var.X)


The first rule aggregates all features of neighbors of the central node, and then we combine the aggregated value with
the value of the second rule, which embeds the features of the central node.

2. Higher-Order Neighborhoods Embedding
#######################################

The second concept is to consider not only the direct neighbors but also higher-order neighbors in the computation of
the central node's representation, such as second-order neighbors (neighbors of neighbors), as can be represented as
the following rule:

.. code-block:: Python

    Relation.layer_1(Var.X) <= (
        Relation.layer_0(Var.Z),
        Relation.edge(Var.Y, Var.X),
        Relation.edge(Var.Z, Var.Y),
        Relation.special.alldiff(...),
    )


.. seealso::

    For more information about the special predicate :code:`alldiff`, see :ref:`special-modifier-label`.


3. Combination of Intermediate Representations
##############################################

The last design concept used in the *H2GCN* model is the combination of intermediate representation.
This can also be easily achieved in PyNeuraLogic just by one rule, where we combine all representations of
layers, such as:

.. code-block:: Python

    Relation.layer_final(Var.X) <= (
        Relation.layer_0(Var.X),
        Relation.layer_1(Var.X),
        Relation.layer_2(Var.X),
        Relation.layer_n(Var.X),
    )
