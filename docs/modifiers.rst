Modifiers
=========

Modifiers are optional and alter an atoms' behavior in some way. Currently, there are two following modifiers, which can be chained together:

Hidden Modifier
###############

..
    TODO: Describe

    .. code-block:: Python

        Relation.edge(1, 2),
        Relation.feature(2)[1.0],

    .. code-block:: Python

        Relation.h(Var.X) <= (Relation.feature(Var.Y), Relation.edge(Var.X, Var.Y))


    TODO: Model graph

    .. code-block:: Python

        Relation.h(Var.X) <= (Relation.feature(Var.Y), Relation.private.edge(Var.X, Var.Y))


    TODO: Model graph

.. _special-modifier-label:

Special Modifier
################

The special modifier changes the atom's behavior depending on its predicate name. We can utilize the following special predicates:

- :code:`Relation.special.alldiff`
    A special atom with the :code:`alldiff` predicate ensures that its terms (logic variables) are substituted for different values (unique values). It's also possible to use :code:`...` in place of terms, which is substituted for all variables declared in the current rule - no variable declared in the rule can be substituted for the same value simultaneously.


.. code-block:: Python

    Relation.special.alldiff(Var.X, Var.Y)  # Var.X cannot equal to Var.Y

    # Var.X != Var.Y != Var.Z
    Relation.h(Var.X) <= (Relation.b(Var.Y, Var.Z), Relation.special.alldiff(...))




- :code:`Relation.special.anypred`

- :code:`Relation.special.in`

- :code:`Relation.special.maxcard`

- :code:`Relation.special.true`

- :code:`Relation.special.false`

- :code:`Relation.special.neq`

- :code:`Relation.special.leq`

- :code:`Relation.special.geq`

- :code:`Relation.special.lt`

- :code:`Relation.special.gt`

- :code:`Relation.special.eq`
