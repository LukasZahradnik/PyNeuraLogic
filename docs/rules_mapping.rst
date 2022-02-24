Understanding Rules
===================

In PyNeuraLogic, describing a model slightly differs from conventional deep learning frameworks.
Instead of putting together a sequence of operations and modules operating on tensors, we define only a model template formed from rules operating on `relations`.
This template is then used to produce computation graphs tailored for each input sample.

But how do those rules translate into computations graphs? Let's take a look at a simple example!

.. image:: _static/ruleexample.svg
    :width: 300
    :alt: Simple Graph
    :align: center


Consider the following relatively simple graph with arbitrarily picked node ids. Usually, we would encode the graph either as
an adjacency matrix or two vectors - :code:`[[sources], [destinations]]`.
The latter representation is also possible in PyNeuraLogic, but we will stick with the relational representation and describe graph edges as :code:`R.edge(<source>, <destination>)`, that is:

.. code-block:: Python

    [
        R.edge(3, 1), R.edge(2, 1), R.edge(4, 3), R.edge(4, 1),
        R.edge(1, 3), R.edge(1, 2), R.edge(3, 4), R.edge(1, 4),
    ]


And just like that, we encoded our example graph with bidirectional edges. Now we can define a few templates upon this graph and dive into how they are being translated or mapped into computational graphs.

The Entry Template
******************

.. code-block:: Python

    R.h(V.X) <= R.edge(V.X, V.Y)

The first template is relatively simple; it contains one rule with only one body relation :code:`R.edge(V.X, V.Y)`.
The rule roughly translates into plain English as

    "To compute representation :code:`h` of any entity :code:`X` aggregate all values of relations :code:`R.edge` where holds that the source node is the entity :code:`X`."

So, for example, for a query :code:`R.h(1)`, there are exactly three edge relations that satisfy the template rule -
:code:`R.edge(1, 2)`, :code:`R.edge(1, 3)`, and :code:`R.edge(1, 4)`. So in the end, we end up with a computation graph like the one below.
The graph is propagated from the bottom (input) level up to the output level, which corresponds to our query.

.. image:: _static/rulecomputationgraph.svg
    :alt: Computation graph
    :align: center


.. note::

    Notice that the input value of all edge relations is :code:`1`. This value has been set implicitly because we didn't provide any.


This visualization renders only the graph's structure without specifying any operations on values passed around. Every node in this graph can have its activation function.

There is one "special" node which is the one highlighted with the magenta color. This node is the so-called "aggregation" node that aggregates all of its inputs and produces one output value (by default averaging inputs).

With that knowledge, it is hopefully now clearer that this graph actually corresponds to what we expressed in "plain English" above and roughly to `message passing` often utilized by Graph Neural Networks.


.. note::

    What if we have a node without any edge and want to compute the :code:`R.h`? We will get an error because we cannot satisfy the rule. Later in this tutorial, we will look at solutions to such a scenario.


Multiple Body Relations
***********************

Our first template was very limited in what we were able to express.
We will often find ourselves declaring rules with multiple body relations to capture more complicated problems.
As an example of such template rule, we could introduce node features and the following rule utilizing said features.
To step up the game even more, we will be introducing weights.

.. code-block::

    R.h(V.X)["a": 1,] <= (R.edge(V.X, V.Y)["b": 1,], R.feature(V.Y)["c": 1,])

.. note::

    We used named weights to make how weights are being mapped to the computation graph more evident. We could simply omit names.


Now we will add arbitrary node features to our example (the encoding of the input graph). For simplicity, features will be scalar, for example:

.. code-block::

    R.feature(1)[0.2], R.feature(2)[0.3], R.feature(3)[0.4], R.feature(4)[0.5]


Again, for the same query :code:`R.h(1)` we will end up with the computational graph below.
The last bottom layer expanded with additional inputs (:code:`R.feature`), and weights appeared to corresponding edges.


.. image:: _static/rulecomputationgraph_features.svg
    :alt: Computation graph with features
    :align: center


This graph highlighted a different level - the level of nodes that operates on the whole rule body (this level was also present in the previous example, but it was meaningless since there was only one body relation). So how do those nodes process their inputs - values from body relations? They concatenate those values.

The concatenation is the summation by default, but it can be adjusted. So, for example, the value of the leftmost magenta node will be calculated as follow (again, without any activation functions):

.. code-block::

    value = (0.3 * c) + (1 * b)


Multiple Rules
**************

Now that we understand how multiple relations in the body are handled and how differently substituted bodies are aggregated, we will look at a scenario with two different rules with the same head.

.. code-block::

    R.h(V.X) <= (R.edge(V.X, V.Y), R.feature(V.Y)),
    R.h(V.X) <= R.feature(V.X),


Up until now, nodes were required to have edges; otherwise, the relation :code:`R.h` couldn't be satisfied. With the additional rules, that is not the case anymore - the second rule will be satisfied for any node with features. Let's take a look at how the mapping changed for this template on the query :code:`R.h(1)`

.. image:: _static/rulecomputationgraph_tworules.svg
    :alt: Computation graph with two rules
    :align: center

We introduced the rightmost branch highlighted with the magenta color by adding the second rule. This branch has the same structure as the right one - there is an aggregation node and node that concatenates body relations, but there isn't much to aggregate nor concatenate.

The interesting part here that might be unclear is the behavior of the topmost node that corresponds to the query - how are its two input branches handled? They are concatenated or aggregated - by default, they are summed.


Graph Readout
*************

Up until now, we have been working with queries on top of one entity - node. What if we wanted to compute the value of relation :code:`R.h` for all available nodes and then somehow aggregate them into one value, i.e., do graph readout?

It could done by listing out relations for all nodes in a single body, but in this case, we can leverage yet again the expressiveness of relational learning.
We can just say, "Aggregate all values of relation :code:`R.h` for all entities :code:`X` that satisfy the relation."
We will use a different query, :code:`R.q`, for the readout for this case.

.. code-block::

    R.h(V.X) <= (R.edge(V.X, V.Y), R.feature(V.Y)),
    R.h(V.X) <= R.feature(V.X),
    R.q <= R.h(V.X),


There is not anything really new in the computational graph below. All of the :code:`R.h` nodes will be unfolded into larger subgraphs, e.g., the :code:`R.h(1)` node will be unfolded to the graph from the previous example.

.. image:: _static/rulecomputationgraph_readout.svg
    :alt: Computation graph with two rules
    :align: center


Activation and Aggregation functions
************************************

We talked about different types of functions defaulting to specific functions, such as average, but how can you customize them?

.. code-block:: Python

    R.h(V.X) <= R.edge(V.X, V.Y)

Let's consider the graph/template from the entry (first) example. To change the activation function (e.g., to sigmoid) of the head of the rule, that is, the topmost node, we can simply add the following to the template.

.. code-block::

    R.h / 1 | [Activation.SIGMOID]

.. note::

    The :code:`/ 1` here defines the arity - we can have multiple relations of the same name with different arities and activation functions.


If we would like to change the aggregation function of the rule, e.g., to the max aggregation function and change the activation of the rule nodes (the ones that are input for the rule aggregation node) to, for example, sigmoid, we would have to actually modify the original rule to the following one:

.. code-block:: Python

    (R.h(V.X) <= R.edge(V.X, V.Y)) | [Aggregation.MAX, Activation.SIGMOID]
