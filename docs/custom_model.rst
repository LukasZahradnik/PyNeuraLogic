Model Definition
================

To be able to learn a problem via the PyNeuraLogic library, it is necessary to encode the problem
as a template, queries and examples. A set of queries, together with a learning example of examples,
forms a dataset.

Template
########

The template (:py:class:`~neuralogic.core.template.Template`) is a set of rules that encode the problem's architecture - i.e. the model, which is
essentially equivalent to composing modules in other frameworks, but more versatile. The versatility
is achieved by utilizing lower-level primitives - the rules that can be freely modified.

Interpretation of Rules
***********************

TODO: Understanding rules


Dataset
#######

The dataset instance holds information about the specific problem instance and is divided into two parts - examples and queries.

.. attention::

    In the context of examples and queries, the weights of atoms are, in fact, not learnable parameters but concrete values that serve as feature inputs (examples) or target labels (queries).

    This means that it is not possible to use the dimension value for the weight (value) definition as it does not represent concrete value.


Examples
********

One example describes a specific graph - an instance of the learning problem encoded as ground atoms/facts and rules. An example can be seen as the input to a model defined by a template.

For example, a complete graph with three nodes and some features can be encoded as:

.. code-block:: Python

    from neuralogic.utils.data.dataset import Dataset
    from neuralogic.core import Relation

    dataset = Dataset()

    dataset.add_example([
        Relation.edge(1, 2), Relation.edge(2, 1), Relation.edge(1, 3),
        Relation.edge(3, 1), Relation.edge(2, 3), Relation.edge(3, 2),

        Relation.feature(1)[0],
        Relation.feature(2)[1],
        Relation.feature(3)[-1],
    ])


Queries
*******

Queries are a set of valued ground atoms (facts) that determine what outputs of the defined template are.


We might, for example, want to learn the output of the :code:`Relation.h` for the entity with index :code:`1` to be :code:`0` and then for the entity with index :code:`2` to be :code:`1`, which might be expressed like this:

.. code-block:: Python

    dataset.add_queries([
        Relation.h(1)[0],
        Relation.h(2)[1],
    ])

Queries are not restricted only to one layer (atom) nor output shape. We can mix queries however we like - for example, we can define query :code:`Relation.a[0]` with scalar label and then query :code:`Relation.b[[1, 0, 1]]` with vector label.

.. note::

    Queries are valued ground atoms, but we don't have to define the value explicitly. If the value is not present, the default value (:code:`1.0`) is used as the label. This can be useful for queries outside of training scenarios - where labels are not needed/known.


.. tip::

    If the learning instance (example) does not change and is the same for every query, we can define only one example, and it will be reused for each query.
