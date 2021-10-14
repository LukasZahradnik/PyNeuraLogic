Modifiers
=========

Modifiers are optional and alter an atoms' behavior in some way. Currently, there are two following modifiers, which can be chained together:

Hidden Modifier
###############

TODO: Describe

.. code-block:: Python

    Atom.edge(1, 2),
    Atom.feature(2)[1.0],

.. code-block:: Python

    Atom.h(Var.X) <= (Atom.feature(Var.Y), Atom.edge(Var.X, Var.Y))


TODO: Model graph

.. code-block:: Python

    Atom.h(Var.X) <= (Atom.feature(Var.Y), Atom.private.edge(Var.X, Var.Y))


TODO: Model graph

.. _special-modifier-label:

Special Modifier
################

The special modifier changes the atom's behavior depending on its predicate name. We can utilize the following special predicates:

- :code:`Atom.special.alldiff`
    A special atom with the :code:`alldiff` predicate ensures that its terms (logic variables) are substituted for different values (unique values). It's also possible to use :code:`...` in place of terms, which is substituted for all variables declared in the current rule - no variable declared in the rule can be substituted for the same value simultaneously.


.. code-block:: Python

    Atom.special.alldiff(Var.X, Var.Y)  # Var.X cannot equal to Var.Y

    # Var.X != Var.Y != Var.Z
    Atom.h(Var.X) <= (Atom.b(Var.Y, Var.Z), Atom.special.alldiff(...))




- :code:`Atom.special.anypred`

- :code:`Atom.special.in`

- :code:`Atom.special.maxcard`

- :code:`Atom.special.true`

- :code:`Atom.special.false`

- :code:`Atom.special.neq`

- :code:`Atom.special.leq`

- :code:`Atom.special.geq`

- :code:`Atom.special.lt`

- :code:`Atom.special.gt`

- :code:`Atom.special.eq`
