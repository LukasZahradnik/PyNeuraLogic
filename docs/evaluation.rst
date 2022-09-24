Model Evaluation
================

Model Building
##############

When we have the template, examples, and queries ready, we need to 'compile' them together to retrieve a model that can be trained and evaluated.

The 'compilation' is done in two steps. Firstly, we retrieve a model instance for the specified backend.

.. code-block:: Python

    from neuralogic.core import Settings

    settings = Settings()
    model = template.build(settings)


Then we can 'build' the examples and queries (dataset), yielding a multitude of computational graphs to be trained.

.. code-block:: Python

    built_dataset = model.build_dataset(dataset)


.. Evaluation
.. ##########

.. TODO


Saving and Loading Model
########################

When our model is trained, or we want to persist the model's state (e.g., make a checkpoint),
we can utilize the model instance method :py:meth:`~neuralogic.nn.base.AbstractNeuraLogic.state_dict` (or :py:meth:`~neuralogic.nn.base.AbstractNeuraLogic.parameters`).
The method puts all parameters' values into a dictionary that can be later saved (e.g., in JSON or in binary) or somehow manipulated.

When we want to load a state into our model, we can then simply pass the state into :py:meth:`~neuralogic.nn.base.AbstractNeuraLogic.load_state_dict` method.

.. note::

    Evaluators offer the same interface for saving/loading of the model.


Utilizing Evaluators
####################

Writing custom training loops and handling different backends can be cumbersome and repetitive. The library offers ‘evaluators’ that encapsulate the training loop and testing evaluation. Evaluators also handle other responsibilities, such as building datasets.

.. code-block:: Python

    from neuralogic.nn import get_evaluator


    evaluator = get_evaluator(template, settings)


Once you have an evaluator, you can evaluate or train the model on a dataset. The dataset doesn't have to be pre-built, as in the case of classical evaluation - the evaluator handles that for you.


.. note::

    If it is used more than once, it is more efficient to pass a pre-built dataset into the evaluator (this will prevent redundant dataset building).


Settings Instance
*****************

The :py:class:`~neuralogic.core.settings.Settings` instance contains all the settings used to customize the behavior of different parts of the library.

Most importantly, it affects the behavior of the model building (e.g., specify default rule/relation transformation functions), evaluators (e.g., error function, number of epochs, learning rate, optimizer),
and the model itself (e.g., initialization of the learnable parameters).

.. code-block:: Python

    from neuralogic.core import Settings, Initializer
    from neuralogic.nn.init import Uniform
    from neuralogic.optim import SGD


    Settings(
        initializer=Uniform(),
        optimizer=SGD(lr=0.1),
        epochs=100,
    )


In the example above, we define settings to ensure that initial values of learnable parameters (of the model these settings are used for) are sampled from the uniform distribution.
We also set properties utilized by evaluators: the number of epochs (:math:`100`) and the optimizer,
which is set to Stochastic gradient descent (SGD) with a learning rate of :math:`0.1`.

Evaluator Training/Testing Interface
************************************

The evaluator's basic interface consists of two methods - :code:`train` and :code:`test` for training on a dataset and evaluating on a dataset, respectively. Both methods have the same interface and are implemented in two modes - generator and non-generator.

The generator mode (default mode) yields a tuple of two elements (total loss and number of instances/samples) per each epoch. This mode can be useful when we want to, for example, visualize, log or do some other manipulations in real-time during the training (or testing).

.. code-block:: Python

    for total_loss, seen_instances in neuralogic_evaluator.train(dataset):
        pass


The non-generator mode, on the other hand, returns only a tuple of metrics from the last epoch.

.. code-block:: Python

    results = neuralogic_evaluator.train(dataset, generator=False)
