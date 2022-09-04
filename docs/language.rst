PyNeuraLogic Language
=====================

The main feature of the PyNeuraLogic library is its custom declarative language
(based on `NeuraLogic <https://github.com/GustikS/NeuraLogic>`_) for describing the structure of the learning problems, data and models.
In PyNeuraLogic, the language is fully embedded in Python, enabling users to utilize Python's convenient modules and features.

The idea of using a custom language (following the logic programming paradigm) instead of the predefined modules, as common in popular
frameworks, is to achieve higher expressiveness, reduce the complexity of writing novel model architectures, and
reveal the underlying relational principles of the models.

This section introduces users to the language's basic syntax, which is essential for understanding concepts presented
in other sections and using the library to its full potential.


Relations
######################

*Relations* are fundamental building blocks of the PyNeuraLogic language. Each instance of a relation consists of four parts - *predicate* name,
an arbitrary number of *terms*, optional *weight* (or *value*), and optional modifier. Predicate name, together
with the "arity" (number of terms) the relation forms its unique signature.

.. image:: _static/atom.svg
    :width: 500
    :alt: Relation structure
    :align: center

|

Relations are created via object :code:`Relation` that can be imported from :code:`neuralogic.core`.


.. tip::

    You can also create relations via :code:`R` object, which is a shortcut of :code:`Relation`.


Predicate name
**************

Predicates serve as a descriptive name for the relations. Predicate names are *case-sensitive* and have to
start with a *lower-case* letter. Usually, relations with specific predicate names are created directly via :code:`Relation` object
(e.g., :code:`Relation.my_rel` creates a relation with the predicate name :code:`my_rel`).
For convenience, we can also use the :code:`Relation.get` method (e.g., :code:`Relation.get("my_rel")`),
which can be useful for generating relations.

.. code-block:: Python

    from neuralogic.core import Relation

    Relation.my_rel  # Relation with a predicate name "my_rel"

    for i in range(5):
        # Relations with predicate names "my_rel_0", ..., "my_rel_4"
        Relation.get(f"my_rel_{i}")


.. note::

    Prepending the predicate name with an underscore (:code:`_`) will make the relation "hidden" (e.g., :code:`Relation.hidden.my_rel` is equal to :code:`Relation._my_rel`). You can read more about modifiers, such as "hidden", in the :ref:`modifier-label` section.


Terms
*****

Terms are an optional list of *constants* and/or logic *variables*.

- **Constants** are either numeric values (floats, integers) or string values with a lower-cased first letter. We can also define a constant via :code:`neuralogic.core.Term`, which converts the provided value into a valid constant (string) for us.

.. code-block:: Python

    from neuralogic.core import Term, Relation

    Relation.my_rel  # A relation with NO terms, also called a "proposition" in logic
    Relation.my_rel(1.0)  # A relation with one constant term 1.0
    Relation.my_rel(Term.my_term, "string_term")  # A relation with two constant terms "my_term" and "string_term"
    Relation.my_rel(1.0, Term.My_Term)   # A relation with two constant terms 1.0 and "my_term"

- **Variables** are *capitalized* string values. We can, similarly to constants, utilize helper :code:`neuralogic.core.Var`, which converts the provided value into a valid variable (string) for us.

.. code-block:: Python

    from neuralogic.core import Var, Relation

    Relation.my_rel(Var.X)  # A relation with one variable "X"
    Relation.my_rel(Var.x, "Y")  # A relation with two variable terms "X" and "Y"

Relations with logical variables express general *patterns*, which is essential for encoding deep *relational* models, such as GNNs.

.. NOTE::
        We call relation "ground" if all of its terms are constants (no variables). These are essentially specific (logical) statements, or *facts*, commonly used to encode the data and particular observations.

Weights
*******

On top of classic relational logic programming, in PyNeuraLogic, the relations can be additionally associated with *weights*.
A relation's weight is optional and servers as a learnable parameter. The weight itself can be defined in the following ways:

- Scalar value defining a learnable scalar parameter initialized to a specific value.

.. code-block:: Python

    Relation.my_rel[0.5]  # Scalar weight initialized to 0.5

- Vector value defining a learnable vector parameter initialized to a specific value.

.. code-block:: Python

    Relation.my_rel[[1.0, 0.0, 1.0]]  # Vector weight initialized to [1.0, 0.0, 1.0]

- Matrix value defining a learnable matrix parameter initialized to a specific value.

.. code-block:: Python

    Relation.my_rel[[[1, 0], [0, 1]]]  # Matrix weight initialized to [[1, 0], [0, 1]]


.. tip::
        Matrix and vector values can also be in the form of `NumPy <https://numpy.org/>`_ arrays.


Instead of defining particular values for the parameters, we can also choose to specify merely the dimensionality of it instead. Here, each element of the parameter represents the size of the corresponding dimension. The initialization of the values in this case is sampled from a distribution determined by the :py:class:`~neuralogic.core.settings.Settings` object.

.. code-block:: Python

    Relation.my_rel[2,]  # Specification of a randomly initialized weight vector of length 2
    Relation.my_rel[3, 3]  # Specification of a randomly initialized 3x3 weight matrix


.. WARNING::
    Notice the difference between :code:`Relation.my_rel[2]` and :code:`Relation.my_rel[2,]` where the first one represents a particular scalar weight with *value* "2", while the latter represents a randomly initialized weight vector of *length* 2.

Named Weights
^^^^^^^^^^^^^

Weight sharing is at the heart of modelling with PyNeuraLogic, where all the (ground) instances of a relation will share its associated parameters. However, you can also choose to share a single weight across multiple relations. This can be achieved by labeling the weight with some name, such as:

.. code-block:: Python

    # Sharing a weight (2x2 matrix weight)
    Relation.my_rel["shared_weight": 2, 2]
    Relation.another_rel["shared_weight": 2, 2]

    # Sharing a weight (vector weight)
    Relation.my_rel["my_weight": 2,]
    Relation.another_rel["my_weight": 2]


Modifiers
*********

Predicate names are generally arbitrary, with no particular meaning other than the user-defined one.
However, by including a modifier in the definition of a relation, we may utilize some of the extra pre-defined predicates with special built-in functionality.

More about individual modifiers can be read in :ref:`modifier-label`.


Rules
#####################


.. code-block:: Python

    Relation.h <= (Relation.b_one, Relation.b_n)


Rules are the core concept in PyNeuraLogic for describing the architectures of the models by defining *templates* for their computational graphs.
Each rule consists of two parts - the *head* and the *body*. The head is an arbitrary relation followed by an implication (:code:`<=`) and subsequently the body formed from a tuple of :code:`n` relations.

When there is only one relation in the body, we can omit the tuple and insert the relation directly.

.. code-block:: Python

    Relation.h <= Relation.b


Such a rule can be then read as *"The relation (proposition) 'h' is implied by the relation (proposition) 'b'"*

Metadata
********
The rules have some (default) properties that influence their translation into the computational graphs (models), such as transformation and aggregation functions.
These properties can be modified, per rule, by attaching a :py:class:`~neuralogic.core.constructs.metadata.Metadata` instance to the rule.

.. code-block:: Python

    from neuralogic.core import Metadata, Transformation, Aggregation


    (Relation.h <= (Relation.b_one, Relation.b_n)) | Metadata(transformation=Transformation.RELU, aggregation=Aggregation.AVG)

    # or, for short, just
    (Relation.h <= (Relation.b_one, Relation.b_n)) | [Transformation.RELU, Aggregation.AVG]


For example, with the construct above, we created a new rule with a specified transformation function (relu) and aggregation function (avg).
