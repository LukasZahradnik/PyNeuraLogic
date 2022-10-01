PyTorch Integration
===================

There are some use cases where we might want to utilize the power of PyTorch, e.g., for some computer vision tasks, and
interconnect PyTorch modules with a relational logic program written in PyNeuraLogic's highly expressive language.

For those scenarios, there is :py:class:`~neuralogic.nn.torch_function.NeuraLogic` as a PyTorch function/module, which
can be used as any other module, such as :code:`torch.nn.Linear`, etc.


Hands-on example
****************

Let's showcase the integration in a simple example. Consider a task to learn the xor function (that we used in different
`examples <https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/SimpleXOR.ipynb>`_ as well). One possible
model written in PyTorch could look like the following:

.. code-block:: python

    import torch

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8, bias=False),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1, bias=False),
        torch.nn.Tanh(),
    )

With the input data :code:`xs` and target labels :code:`ys`:

.. code-block:: python

    xs = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=torch.float32)

    ys = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


But what if, for whatever reason, we would want to replace the second :code:`torch.nn.Linear` layer with an alternative expressed
in PyNeuraLogic? We would proceed as usual with a definition of a template. In our case, it is a simple one-rule
template representing a linear layer.


.. code-block:: python

    from neuralogic.core import R, Template
    import neuralogic.nn.functional as F


    template = Template()

    template += (R.xor[1, 8] <= R.xy) | [F.identity]
    template += R.xor / 0 | [F.identity]


The next step is to describe how should be the output of preceding torch layers (in the form of tensors) mapped into
NeuraLogic. For those purposes, we declare a function :code:`to_logic` that will take all arguments
(:code:`args` and :code:`kwargs`) passed into the forward function of the
:py:class:`~neuralogic.nn.torch_function.NeuraLogic` module and will assign values to facts in our model. In our case,
it will be only one tensor - the output from the first :code:`torch.nn.Tanh`, that we will assign as a value to
the :code:`R.xy` fact.

.. code-block:: python

    def to_logic(tensor_input):
        return [
            R.xy[tensor_input],
        ]



We can now put it all together and replace the PyTorch linear layer with our NeuraLogic linear layer by initializing
the :py:class:`~neuralogic.nn.torch_function.NeuraLogic` module (extends :code:`torch.nn.Module`). The module requires
arguments, that is, the template, the example with facts and their initial values (in our case, it is only
:code:`R.xy` with a random vector value of length 8 - the size of the output of the previous layer),
the query (output predicate), and the mapping function.

.. note::

    We have onest atic computation graph and change input facts values here, in contrast to the usual NeuraLogic
    workflow, where we have a different computation graph for each example-query set.


.. code-block:: python

    from neuralogic.nn.torch_function import NeuraLogic


    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8, bias=False),
        torch.nn.Tanh(),
        NeuraLogic(template, [R.xy[8,]], R.xor, to_logic),
        torch.nn.Tanh(),
    )


We can now create a classic training loop, similarly as you might do in the case of pure PyTorch.

.. note::

    Currently, the torch optimizer is not connected to the NeuraLogic module. Weights updates of the NeuraLogic module
    will be done during the backward propagation instead of on the :code:`optimizer.step` call.

    In addition, :code:`model.parameters()` will not contain actual parameters of the NeuraLogic module - you can access
    them via :code:`NeuraLogic(...).model.parameters()`, similarly you can load state via
    :code:`NeuraLogic(...).model.load_state_dict`.

.. code-block:: python

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = torch.nn.MSELoss()


    for _ in range(400):
        for x, y in zip(xs, ys):
            output = model(x)
            loss_value = loss(output, y)

            optimizer.zero_grad(set_to_none=True)
            loss_value.backward()
            optimizer.step()


    for x in xs:
        print(model(x))


.. code-block:: python

    tensor(0., grad_fn=<TanhBackward0>)
    tensor(0.8837, grad_fn=<TanhBackward0>)
    tensor(0.8738, grad_fn=<TanhBackward0>)
    tensor(0.0245, grad_fn=<TanhBackward0>)
