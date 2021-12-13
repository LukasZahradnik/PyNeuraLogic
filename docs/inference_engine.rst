Utilizing Inference Engine
==========================

While translating logic programs into computations graphs, PyNeuraLogic utilizes an `inference engine <https://en.wikipedia.org/wiki/Inference_engine>`_.
The inference engine serves for deducing information from the input knowledge base encoded in examples or a template.
For convenience, this functionality is also exposed via a high-level interface to be accessible for users.

London Underground Example
##########################

The interface for the inference engine is relatively simple. Consider the following example based on `the "Simply Logical: Intelligent Reasoning by Example" book <https://book.simply-logical.space/>`_ by Peter Flach.
We have a part of the London Underground encoded as a directed graph as visualized in the following image.


|

.. image:: _static/london.svg
    :height: 100
    :alt: London Underground
    :align: center

|


This graph can be encoded as :code:`connected(From, To, Line)` such as:

.. code-block:: Python

    from neuralogic.core import Template, R, V, T


    template = Template()
    template += [
        R.connected(T.bond_street, T.oxford_circus, T.central),
        R.connected(T.oxford_circus, T.tottenham_court_road, T.central),
        R.connected(T.bond_street, T.green_park, T.jubilee),
        R.connected(T.green_park, T.charing_cross, T.jubilee),
        R.connected(T.green_park, T.piccadilly_circus, T.piccadilly),
        R.connected(T.piccadilly_circus, T.leicester_square, T.piccadilly),
        R.connected(T.green_park, T.oxford_circus, T.victoria),
        R.connected(T.oxford_circus, T.piccadilly_circus, T.bakerloo),
        R.connected(T.piccadilly_circus, T.charing_cross, T.bakerloo),
        R.connected(T.tottenham_court_road, T.leicester_square, T.northern),
        R.connected(T.leicester_square, T.charing_cross, T.northern),
    ]


This template essentially encodes only direct connections between stations (nodes).
We might want to extend this knowledge by deducing which stations are nearby - stations with at most one station between them.

So stations are nearby if they are directly connected, which can be expressed as:

.. code-block:: Python

    template += R.nearby(V.X, V.Y) <= R.connected(V.X, V.Y, V.L)

Stations are also nearby if exactly one station lays on the path between those two stations and are on the same line.

.. code-block:: Python

    template += R.nearby(V.X, V.Y) <= (R.connected(V.X, V.Z, V.L), R.connected(V.Z, V.Y, V.L))


Now we can ask the inference engine to get all sorts of different information, such as what stations the Oxford Circus station is nearby.

.. code-block:: Python

    from neuralogic.core.inference_engine import InferenceEngine


    engine = InferenceEngine(template)

    engine.q(R.nearby(V.X, T.oxford_circus))

Running the :code:`query` (or :code:`q`) will return a generator of dictionaries with all possible substitutions for all variables in the query.
In this case, we have only one variable in the query (:code:`V.X`). As you can see, the inference engine found all stations that the Oxford Circus station is nearby (Green Park and Bond Street).

.. code-block::

    [
        {"X": "green_park"},
        {"X": "bond_street"},
    ]

We could also ask the inference engine to get all possible nearby stations (:code:`R.nearby(V.X, V.Y)`) and so on.


Finding Path Recursively
************************

We can also define another rule to check a generic path from some station :code:`X` to some station :code:`Y`.
We will call this rule reachable and use recursion in its definition. The reachable rule is satisfactory if two stations are directly connected or station :code:`X` is connected to station :code:`Z` from which you can reach :code:`Y`.

.. code-block:: Python

    template += R.reachable(V.X, V.Y) <= R.connected(V.X, V.Y, V.L)
    template += R.reachable(V.X, V.Y) <= (R.connected(V.X, V.Z, V.L), R.reachable(V.Z, V.Y))

Now we can ask the inference engine what stations we can reach from some station or ask more exact queries such as if two specific stations are reachable.


.. code-block:: Python

    engine = InferenceEngine(template)

    if engine.query(R.reachable(T.green_park, T.tottenham_court_road)):
        print("Yes, you can reach Tottenham Court Road from Green Park")
    else:
        print("Those two stations are reachable, so this should never be printed out")
