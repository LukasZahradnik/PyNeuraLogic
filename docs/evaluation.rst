Model Evaluation
================

Model Building
##############

When we have the template, examples, and queries ready, we need to 'compile' them together to retrieve a model that can be trained and evaluated.

The 'compilation' is done in two steps. Firstly, we retrieve a model instance for the specified backend.

.. code-block:: Python

    model = template.build()


Then we can 'build' the examples and queries (dataset), yielding a multitude of computational graphs to be trained.

.. code-block:: Python

    built_dataset = model.build_dataset(dataset)


.. Evaluation
.. ##########

.. TODO


Saving and Loading Model
########################

When our model is trained, or we want to persist the model's state (e.g., make a checkpoint),
we can utilize the model instance method :py:meth:`~neuralogic.core.neural_module.NeuralModule.state_dict` (or :py:meth:`~neuralogic.core.neural_module.NeuralModule.parameters`).
The method puts all parameters' values into a dictionary that can be later saved (e.g., in JSON or in binary) or somehow manipulated.

When we want to load a state into our model, we can then simply pass the state into :py:meth:`~neuralogic.core.neural_module.NeuralModule.load_state_dict` method.

.. note::

    Evaluators offer the same interface for saving/loading of the model.


Settings Instance
*****************

The :py:class:`~neuralogic.core.settings.Settings` instance contains all the settings used to customize the behavior of different parts of the library.

Most importantly, it affects the behavior of the model building (e.g., specify default rule/relation transformation functions), evaluators (e.g., error function, number of epochs, learning rate, optimizer),
and the model itself (e.g., initialization of the learnable parameters).

.. code-block:: Python

    from neuralogic import Settings
    from neuralogic.nn.init import Uniform
    from neuralogic.nn.optim import SGD


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

The model's basic interface consists of two methods — :code:`train` and :code:`test` for training on a dataset and evaluating on a dataset, respectively.

The :code:`train` method accepts a dataset and a number of epochs, runs the training loop, and returns the training results — a list of :code:`(target, output, error)` tuples (or a single such tuple for a single sample).

.. code-block:: Python

    results = model.train(dataset, epochs=100)

The :code:`test` method accepts a dataset and returns the model outputs — a list of output values (or a single value for a single sample).

.. code-block:: Python

    outputs = model.test(dataset)
