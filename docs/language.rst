PyNeuraLogic Language
=====================

Additionally to predefined modules, PyNeuraLogic allows users to encode machine learning problems via parameterized,
rule-based constructs. Said constructs are based on a custom declarative language [Neuralogic]_ that follows a logic programming paradigm.



The anatomy of a rule
#####################

In PyNeuraLogic, rules are primitives used for building models and datasets.

.. code-block:: Python

    from neuralogic.core import Atom, Var


    Atom.h(Var.X)[W_0] <= (Atom.feature(Var.Y)[W_1], Atom.edge(Var.X, Var.Y))

The rule consists of a head (:code:`Atom.h`) and a body (:code:`Atom.feature`, :code:`Atom.edge`). Our example can be
then read as:
    "Atom h is implied by atom feature and atom edge"




.. [Neuralogic] https://github.com/GustikS/NeuraLogic
