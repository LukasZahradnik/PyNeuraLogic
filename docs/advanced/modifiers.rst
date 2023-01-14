.. _modifier-label:

Modifiers
=========

Modifiers are optional and alter an relations' behavior in some way. Currently, there are two following modifiers, which can be chained together:

Hidden Modifier
###############

Sometimes, there are relations in rules that only define the logic structure and are not beneficial to be included in the
computation graph. For those cases, there is a hidden modifier that enforces exactly that -
includes relation for the logic part and excludes relation in the resulting computation graph.

For example, consider the following rule. In some instances, it might be counterproductive to include the :code:`edge` relations
in the resulting computation graph (e.g., they might not have any edge features), yet those :code:`edge` relations
cannot be removed as they define a critical part of the logic structure of the program.
Including them in the computation graph will produce a side effect - offsetting the result of relations :code:`h`.

.. code-block:: Python

    Relation.h(Var.X) <= (Relation.feature(Var.Y), Relation.edge(Var.X, Var.Y))

This issue can be solved by flagging the predicate :code:`edge` as :code:`hidden`, ensuring that relations with such a predicate will not be included in the computation graph.


.. code-block:: Python

    Relation.h(Var.X) <= (Relation.feature(Var.Y), Relation.hidden.edge(Var.X, Var.Y))

    # can be written also as (prepended _ makes predicate hidden)

    Relation.h(Var.X) <= (Relation.feature(Var.Y), Relation._edge(Var.X, Var.Y))


.. _special-modifier-label:

Special Modifier
################

The special modifier changes the relation's behavior depending on its predicate name. We can utilize the following special predicates:

- :code:`Relation.special.alldiff`
    A special relation with the :code:`alldiff` predicate ensures that its terms (logic variables) are substituted for different values (unique values). It's also possible to use :code:`...` in place of terms, which is substituted for all variables declared in the current rule - no variable declared in the rule can be substituted for the same value simultaneously.


.. code-block:: Python

    Relation.special.alldiff(Var.X, Var.Y)  # Var.X cannot equal to Var.Y

    # Var.X != Var.Y != Var.Z
    Relation.h(Var.X) <= (Relation.b(Var.Y, Var.Z), Relation.special.alldiff(...))


- :code:`Relation.special.next`

- :code:`Relation.special.anypred`

- :code:`Relation.special._in`

- :code:`Relation.special.maxcard`

- :code:`Relation.special.true`

- :code:`Relation.special.false`

- :code:`Relation.special.neq`

- :code:`Relation.special.leq`

- :code:`Relation.special.geq`

- :code:`Relation.special.lt`

- :code:`Relation.special.gt`

- :code:`Relation.special.eq`

- :code:`Relation.special.add`

- :code:`Relation.special.sub`

- :code:`Relation.special.mod`
