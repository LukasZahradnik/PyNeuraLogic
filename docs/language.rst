PyNeuraLogic Language
=====================

The main feature of the PyNeuraLogic library is its custom declarative language
(based on `NeuraLogic <https://github.com/GustikS/NeuraLogic>`_) for describing problems' architectures and data.
The language is fully implemented in Python, enabling users to utilize Python's convenient modules and features.

The idea of using a custom language (based on logic programming paradigm) instead of predefined modules as in popular
frameworks and libraries is to achieve higher expressiveness, reduce the complexity of writing models' architectures and
unfold models' architectures from black-box representations.

This section introduces users to the language's basic syntax, which is essential for understanding concepts presented
in other sections and using the library to its full potential.


The Anatomy of an Atom
######################

Atoms are fundamental building blocks of the PyNeuraLogic language. An atom consists of four parts - predicate name,
an arbitrary number of terms, optional weight (in some cases value), and optional modifier. Predicate name, together
with atom's arity (number of its terms), uniquely identifies an atom.

.. image:: _static/atom.svg
    :width: 500
    :alt: Atom structure
    :align: center

|

Atoms are created via object :code:`Atom` (instance of :code:`AtomFactory`) that can be imported from
:code:`neuralogic.core`.

Predicate name
**************

Predicate can serve as a descriptor of data that the atom represents. Predicate names are case-sensitive and have to
start with a lower-case letter. Usually, atoms with specific predicate names are created directly via :code:`Atom` object
(e.g., :code:`Atom.my_atom` creates an atom with the predicate name :code:`my_atom`).
For convenience, we can also use the :code:`Atom.get` method (e.g., :code:`Atom.get("my_atom")`),
which can be useful for generating atoms.

.. code-block:: Python

    from neuralogic.core import Atom

    Atom.my_atom  # Atom with a predicate name "my_atom"

    for i in range(5):
        # Atoms with a predicate names "my_atom_0", ..., "my_atom_4"
        Atom.get(f"my_atom_{i}")


Terms
*****

Terms are an optional list of constants and logic variables.

- Constants are either numeric values (floats, integers) or string values with a first lower-cased letter. We can also define constant via :code:`neuralogic.core.Term`, which converts the provided value into a valid constant (string) for us.

.. code-block:: Python

    from neuralogic.core import Term, Atom

    Atom.my_atom(1.0)  # Atom with one constant term 1.0
    Atom.my_atom(Term.my_term, "string_term")  # Atom with two constant terms "my_term" and "string_term"
    Atom.my_atom(1.0, Term.My_Term)   # Atom with two constant terms 1.0 and "my_term"

- Variables are capitalized string values. We can, similarly to constants, utilize helper :code:`neuralogic.core.Var`, which converts the provided value into a valid variable (string) for us.

.. code-block:: Python

    from neuralogic.core import Var, Atom

    Atom.my_atom(Var.X)  # Atom with one variable "X"
    Atom.my_atom(Var.x, "Y")  # Atom with two variable terms "X" and "Y"


Weights
*******

Atom's weight is optional and defines the atom's learnable parameter. The weight itself can be defined in the following ways:

- The scalar value defines one learnable scalar parameter initialized to the specific value.

.. code-block:: Python

    Atom.my_atom[0.5]  # Scalar weight initialized to 0.5

- The vector value defines the learnable vector parameter initialized to the specific value.

.. code-block:: Python

    Atom.my_atom[[1.0, 0.0, 1.0]]  # Vector weight initialized to [1.0, 0.0, 1.0]

- The matrix value defines the learnable matrix parameter initialized to the specific value.

.. code-block:: Python

    Atom.my_atom[[[1, 0], [0, 1]]]  # Matrix weight initialized to [[1, 0], [0, 1]]


.. NOTE::
        Matrix and vector values can also be in the form of `NumPy <https://numpy.org/>`_ arrays.

- The dimension value is represented as a tuple of either one or two elements. Each element represents the size of one dimension; thus, it can represent either vector or matrix. The difference between previous representations is that the dimension value is less verbose and doesn't describe the initialized value of the parameter - the initialization of dimension values is determined by the settings object.

.. code-block:: Python

    Atom.my_atom[2,]  # Dimension weight representing vector of length of 2
    Atom.my_atom[3, 3]  # Dimension weight representing 3x3 matrix


.. WARNING::
    Notice the difference between :code:`Atom.my_atom[2]` and :code:`Atom.my_atom[2,]` as the first one represents the scalar weight and the latter one dimension (vector of length of two) weight.

Named Weights
^^^^^^^^^^^^^

In case we want to share one weight for multiple atoms, we can achieve that by labeling the weight with an arbitrary name, such as:

.. code-block:: Python

    # Sharing dimension weight (2x2 matrix weight)
    Atom.my_atom["shared_weight": 2, 2]
    Atom.another_atom["shared_weight": 2, 2]

    # Sharing dimension weight (vector weight)
    Atom.my_atom["my_weight": 2,]
    Atom.another_atom["my_weight": 2]


.. _modifier-label:

Modifiers
*********

Modifiers are optional and alter an atoms' behavior in some way. Currently, there are two following modifiers, which can be chained together:

Hidden Modifier
^^^^^^^^^^^^^^^

TODO: Describe


.. _special-modifier-label:

Special Modifier
^^^^^^^^^^^^^^^^

The special modifier changes the atom's behavior depending on its predicate name. We can utilize the following special predicates:

- :code:`Atom.special.alldiff`
    A special atom with the :code:`alldiff` predicate ensures that its terms (logic variables) are substituted for different values (unique values). It's also possible to use :code:`...` in place of terms, which is substituted for all variables declared in the current rule - no variable declared in the rule can be substituted for the same value simultaneously.


.. code-block:: Python

    Atom.special.alldiff(Var.X, Var.Y)  # Var.X cannot equal to Var.Y

    # Var.X != Var.Y != Var.Z
    Atom.h(Var.X) <= (Atom.b(Var.Y, Var.Z), Atom.special.alldiff(...))


The Anatomy of a Rule
#####################

- Rule example
