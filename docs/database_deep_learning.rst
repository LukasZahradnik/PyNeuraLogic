Deep Learning on Databases
==========================


Before running machine learning models written in some popular frameworks on data coming from relational databases, we
must first figure out how to transform data properly. Usually, models expect input data to be in the form of fixed-size
numeric tensors. Making such a transformation on one database table (relation) might be straightforward. However,
it does not have to be the case when our data are linked via various (attributed) relations, which is not very unusual.

So is there a way to treat relational data more simply and efficiently? In the NeuraLogic language, based on relational
logic, relations are *first-class citizens*. Everything is built around and upon them, making feature engineering of
relational data (such as from relational databases) *intuitive* and *direct*.

Since relational databases are such an essential topic that most developers and engineers come in touch with on a daily
basis, the *PyNeuraLogic* framework is equipped with a set of tools to handle loading data from databases and exporting
your trained model to SQL so that you can evaluate your models directly in your databases!

Let's dive into an example where we explore ways of loading data from a relational database, training a model on data,
and exporting the trained model to SQL.

.. note::
    :class: empty-title, outline-only

    ⚗️ Evaluation of trained models directly in the database (conversion to SQL) is currently an experimental feature;
    therefore, it comes with a few restrictions - models cannot contain recursive rules, matrices, and vectors.
    Also, only a limited set of activation functions is implemented.


Data loading from database
**************************

.. note::
    :class: empty-title

    💾 `Download database tables SQL dump. <https://gist.github.com/LukasZahradnik/76b8b6b5e1d6f20752fdb8c61098b21b>`_


First things first, get to know the data we work with. Our example database contains information about molecules. Each
molecule has some attributes and is formed by various numbers of atoms. Atoms also have some attributes and can be in
bond (connected) with another atom. Therefore, our database consists of three tables - *molecule*, *atom*, and *bond*.

.. figure:: _static/mutagenesis_tables.svg
    :width: 500
    :alt: Mutagenesis DB tables
    :align: center

    Database diagram of the molecule database we are working with


In this scenario, our task is to determine the mutagenicity of a molecule on Salmonella typhimurium. Our target is the
*mutagenic* field in the *molecule* table.

But how exactly can we transform those tabular data to *something* that PyNeuraLogic would understand, that is,
relations? We have to define mappings according to our needs.
For example, we would want to map each row of the *atom* table to the relation *R.atom* with *atom_id* and *molecule_id*
fields as the relations' terms and the *charge* field as the relations' values.

.. figure:: _static/mutagenesis_table_mapping.svg
    :width: 500
    :alt: Mapping DB table to PyNeuraLogic relations
    :align: center

    Mapping of one table to valued facts. We map *atom_id* and *molecule_id* columns to terms the *charge*
    column to fact value.


Notice that in the mapping visualization above, we skipped *element* and *type* fields. That is because we will
not utilize them in our model, but there is nothing stopping us from mapping all fields to terms or even mapping
one database table to multiple relations.

The mapping itself is quite straightforward; we create an instance of *DBSource* with *relation name*,
*table name*, and *column names* (that will be mapped to terms) as its arguments.
We will utilize only data from *bond* and *atom* tables in our example set to keep it simple.

.. code-block:: python

    from neuralogic.dataset.db import DBSource

    atoms = DBSource("atom", "atom", ["atom_id", "molecule_id"], value_column="charge")
    bonds = DBSource("bond", "bond", ["atom1_id", "atom2_id", "type"], default_value=1)

To train our model, we also need labels. We can find them in the molecule table under the mutagenic field. But this
field contains textual data ("yes"/"no"), so we cannot just simply load the column as values (labels) of queries;
we have to do a little bit of postprocessing. For those scenarios, *DBSource* can take a *value_mapper* argument
that maps the original value from a table to some arbitrary numeric value.

.. code-block:: python

    queries = DBSource(
        "mutagenic",
        "molecule",
        ["molecule_id"],
        value_column="mutagenic",
        value_mapper=lambda value: 1 if value == "yes" else 0
    )

Since our task is to determine the mutagenicity, let's give our queries proper naming, i.e., *mutagenic*,
that is more self-explaining (and can be more understandable by other team members). Let's put everything together and
create a connection with some compatible driver (such as psycopg2 or MariaDB) and create a logic dataset.
With just those few lines of code, we have managed to create a dataset in the logic representation (relations)
populated from a database.

.. code-block:: python

    import psycopg2

    with psycopg2.connect(**connection_config) as connection:
        dataset = DBDataset(connection, [bonds, atoms], queries)
        logic_dataset = dataset.to_dataset()


Training on data from database
******************************

The dataset is ready; let's take a look at defining a template. A template can be seen as a high-level blueprint for
constructing a computation graph tailored for each query (sample).

The template we define contains embeddings for each type of bond (bond type is an integer in the range 1-7). Then we
define four stacked *Message Passing Neural Networks* (*MPNNs*) where edges are bonds and nodes are atoms. Our proposed
layers are similar to the *GraphSAGE* architecture except for extra edge (*bond*) embeddings. The template then
defines a readout layer (*mutagenic*) that pools embeddings of all nodes from all layers and aggregates them into one
value passed into a sigmoid function.

.. code-block:: python

    from neuralogic.core import Template, R, V, Activation


    template = Template()
    template += [R.bond_embed(bond_type)[1,] for bond_type in range(1, 8)]

    template += R.layer1(V.A)[1,] <= (R.atom(V.N, V.M)[1,], R.bond_embed(V.B)[1,], R._bond(V.N, V.A, V.B))
    template += R.layer1(V.A)[1,] <= R.atom(V.A, V.M)[1,]

    template += R.layer2(V.A)[1,] <= (R.layer1(V.N)[1,], R.bond_embed(V.B)[1,], R._bond(V.N, V.A, V.B))
    template += R.layer2(V.A)[1,] <= R.layer1(V.A)[1,]

    template += R.layer3(V.A)[1,] <= (R.layer2(V.N)[1,], R.bond_embed(V.B)[1,], R._bond(V.N, V.A, V.B))
    template += R.layer3(V.A)[1,] <= R.layer2(V.A)[1,]

    template += R.layer4(V.A)[1,] <= (R.layer3(V.N)[1,], R.bond_embed(V.B)[1,], R._bond(V.N, V.A, V.B))
    template += R.layer4(V.A)[1,] <= R.layer3(V.A)[1,]

    template += (R.mutagenic(V.M)[1,] <= (
        R.layer1(V.A)[1,], R.layer2(V.A)[1,], R.layer3(V.A)[1,], R.layer4(V.A)[1,], R.atom(V.A, V.M)[1,]
    )) | [Activation.IDENTITY]

    template += R.mutagenic / 1 | [Activation.SIGMOID]


Now we can build our model by passing the template into an evaluator. We then use the evaluator
to train the model on our dataset.

.. code-block:: python

    from neuralogic.nn import get_evaluator
    from neuralogic.core import Settings, Optimizer
    from neuralogic.nn.init import Glorot
    from neuralogic.nn.loss import CrossEntropy


    settings = Settings(
        optimizer=Optimizer.ADAM, epochs=2000, initializer=Glorot(), error_function=CrossEntropy(with_logits=False)
    )

    neuralogic_evaluator = get_evaluator(template, settings)
    built_dataset = neuralogic_evaluator.build_dataset(logic_dataset)

    for epoch, (total_loss, seen_instances) in enumerate(neuralogic_evaluator.train(built_dataset)):
        print(f"Epoch {epoch}, total loss: {total_loss}, average loss {total_loss / seen_instances}")


Converting model to SQL
***********************

With just a few lines of code, the model that we just built from scratch and trained can be turned into (Postgres) SQL
code. By doing so, you can evaluate the model directly on your database server without installing *NeuraLogic* or even
Python. Just plain *PostgreSQL* will do!

All we have to do is create a converter that takes our model, table mappings, and settings. Table mappings are similar
to *DBSource* from the beginning of this article and map relation name to table name and terms to
column names in the table.


.. code-block:: python

    from neuralogic.db import PostgresConverter, TableMapping


    convertor = PostgresConverter(
        neuralogic_evaluator.model,
        [
            TableMapping("_bond", "bond", ["atom1_id", "atom2_id", "type"]),
            TableMapping("atom", "atom", ["atom_id", "molecule_id"], value_column="charge")
        ],
        settings,
    )

Before you can evaluate your model in SQL, it is necessary to do a proper setup. We can achieve this by
installing the SQL code returned from the *get_std_functions* method. This SQL code will create two schemes (namespaces)
- *neuralogic_std* and *neuralogic*, and a minimal set of generic functions used in your model
(activations, aggregations, etc.) in the prior scheme (e.g., *neuralogic_std.sigmoid*).
The latter scheme will contain the functions for evaluating your model.

.. code-block:: python

    std_sql = convertor.get_std_functions()

After installing the first SQL code, you can install your actual model as SQL code that can be retrieved by calling the
*to_sql* method.

.. code-block:: python

    sql = convertor.to_sql()


.. note::
    :class: empty-title

    💾 `Download the full SQL dump (with std functions) of the trained model. <https://gist.github.com/LukasZahradnik/cb4535a272026f088d60b09e68bc03b3>`_

You are set and ready to evaluate your trained model directly in the database without data ever leaving it.
For each fact and head of a rule, there is a corresponding function in the *neuralogic* namespace.
Let's say we would want to evaluate our model on a molecule with an id *"d150"*.
It is as simple as making one select statement!


.. code-block:: sql

    SELECT * FROM neuralogic.mutagenic('d150');


.. image:: _static/sql_query_result.svg
    :width: 650
    :alt: The result of mutagenic('d150')
    :align: center

|

The evaluation is not limited only to one molecule id. It is possible to use a *NULL* as a "placeholder"
(just like a variable in NeuraLogic) and retrieve all inferrable substitutions and their values.

.. code-block:: sql

    SELECT * FROM neuralogic.mutagenic(NULL);


.. image:: _static/sql_query_results.svg
    :width: 650
    :alt: The results of mutagenic(NULL)
    :align: center

|

As we already said before, every rule head has its corresponding function (if there is no table mapping attached).
This means we can even inspect values of different layers, for example, the value of atom *d15_11* in the first layer.

.. code-block:: sql

    SELECT * FROM neuralogic.layer1('d15_11');

.. image:: _static/sql_layer_query_result.svg
    :width: 650
    :alt: The results of layer1('d15_11')
    :align: center

|


Conclusion
**********
This short tutorial introduced and demonstrated PyNeuraLogic's support of deep learning on databases on a simple example
(learning on molecules). We went through how to fetch data from a database, transform them into PyNeuraLogic
relations with just a few lines of code, and train a model on those data. After training the model, we dumped it
to SQL code, which allowed us to evaluate the model directly in the database.
