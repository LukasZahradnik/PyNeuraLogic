Problem Definition
==================

To approach relational machine learning problems with the PyNeuraLogic library in its full potential, we generally divide each learning scenario into (i) learning examples, (ii) queries, and (iii) a learning template. A set of learning examples together with the queries form a learning dataset. The learning template then constitues a "lifted" model architecture, i.e. a prescription for unfolding (differentiable) computational graphs.

Dataset
#######

The dataset object holds factual information about the problem and is divided into two parts - (i) examples and (ii) queries.

.. attention::

    In the context of examples and queries, the "weights" of the relations are, in fact, not learnable parameters but concrete *values* that serve as inputs (example features) or target outputs (query labels).

    This means that it is not possible to use the dimensionality definition for the weight (value) in this case, as it does not represent a concrete value.


Examples
********

An example describes a specific learning instance, such as a graph, generally encoded through the language of *ground* relations/facts and rules. Intuitively, a learning example can be seen as the input to the model defined by a template.

Examples can be loaded from files in various formats, or encoded directly in Python in the NeuraLogic language.
For instance, a complete graph with three nodes and some features can be encoded as:

.. code-block:: Python

    from neuralogic.core import Relation, Dataset

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

Queries are relations (facts) corresponding to the desired outputs of the learning model/template. These are commonly associated with (non-learnable) weights determining the expected values of the target (relation) labels, given some input example(s).


We might, for example, want to learn the output values of the unary relation (property) :code:`Relation.h` of the entity :code:`anna` to be :code:`0`, and for the entity :code:`elsa` to be :code:`1`. This might be expressed like this:

.. code-block:: Python

    dataset.add_queries([
        Relation.h('anna')[0],
        Relation.h('elsa')[1],
    ])

Note that, in constrast to classic machine learning labels, queries are not restricted to a single target "output" in the template, such as the "output layer" in classic neural models. We can thus ask different completely arbitray queries at the same time:

.. code-block:: Python

    dataset.add_queries([
        Relation.h('anna')[0],
        Relation.h('elsa')[1],
        Relation.friend('anna','elsa')[1],
    ])

Also, the associated labels can be of arbitrary shapes. We can thus, for example, combine a query :code:`Relation.a[0]` with a scalar label with a query :code:`Relation.b[[1, 0, 1]]` with a vector label, each associated with a different part of the learning template.

.. note::

    Queries are valued ground relations, but we don't have to define the value explicitly. If the value is not present, the default value (:code:`1.0`) is used as the label. This is useful, e.g., for queries outside the learning phase, where the labels are not needed/known.


A single learning example may then be associated with a single query, as common in classic supervised machine learning, or with multiple queries, as common e.g. in knowledge-base completion or collective classification tasks.

.. tip::

    If the learning example does not change and is the same for every query, we can simly define only one example, and it will be reused for each query.


Template
########

The template (:py:class:`~neuralogic.core.template.Template`) is a set of *rules* that encode the lifted model architecture. Intuitively, this is somewhat similar to composing modules in the common deep learning frameworks, but more versatile. The versatility follows from the *declarative* nature of the rules, which can be highly abstract and expressive, just like the modules, yet directly reveal an interface to the underlying lower-level principles of the module's computation.
