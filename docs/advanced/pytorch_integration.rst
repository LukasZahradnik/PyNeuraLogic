PyTorch Integration
===================

PyNeuraLogic offers two distinct ways to work with PyTorch:

+----------------------------------------------------+----------------------------------------------------+
| **Core PyTorch Backend**                            | **NeuraLogic ``nn.Module`` Wrapper**               |
+====================================================+====================================================+
| Makes an entire PyNeuraLogic model trainable with   | Wraps a PyNeuraLogic template as a standard        |
| PyTorch optimizers and loss functions.              | ``nn.Module`` layer you can drop into any          |
|                                                    | ``nn.Sequential`` pipeline.                        |
+----------------------------------------------------+----------------------------------------------------+
| Entry point: ``model.build(settings, torch=True)``  | Entry point: ``from neuralogic.nn.torch_function`` |
|                                                    | ``import NeuraLogic``                              |
+----------------------------------------------------+----------------------------------------------------+

The sections below walk through each integration, its constraints, and when to pick it.


Integration 1: Core PyTorch Backend
************************************

Pass :code:`torch=True` when building your model and every learnable weight becomes a
:py:class:`~neuralogic.core.torch.tensor.NeuralogicOptTensor` — a :code:`torch.Tensor` subclass that standard
PyTorch optimizers can update. The forward pass returns an autograd-aware tensor,
so :code:`loss.backward()` and :code:`optimizer.step()` work exactly as you'd expect.

Enabling
--------

.. code-block:: python

    from neuralogic import Model
    from neuralogic.core import Settings

    model = Model()
    # ... add rules and relations ...
    model = model.build(Settings(...), torch=True)

Once built, call :code:`model.tensor_parameters()` to get the list of learnable tensors and pass it to any
:code:`torch.optim` optimizer:

.. code-block:: python

    params = model.tensor_parameters()
    optimizer = torch.optim.Adam(params, lr=1e-3)

Constraints
-----------

**Not a drop-in ``nn.Module`` layer.** The core backend takes PyNeuraLogic datasets (built via
:code:`model.build_dataset(dataset)`) as input — not raw tensors. This means you **cannot** insert a model
built this way into :code:`nn.Sequential` or pass it a plain :code:`torch.Tensor`. If you need a
PyNeuraLogic component in the middle of a pipeline, use :ref:`Integration 2 <pytorch-integration-2>`.

However, you **can** use a torch-backed model as the *first stage* of a pipeline — feed it a dataset,
and pass the resulting tensor into a downstream :code:`nn.Sequential`:

.. code-block:: python

    pyneura_model = model.build(settings, torch=True)
    downstream = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
    )

    # forward
    output = pyneura_model(built_dataset)   # returns a torch.Tensor
    result = downstream(output)             # feeds into standard PyTorch layers

The PyNeuraLogic model must come first because it needs a dataset on input; the downstream layers
consume the tensor it produces.

**One model, one optimizer.** The model *is* your entire architecture. You train it with a single PyTorch
optimizer and loss function — there is no built-in trainer involved.

**Parameter access.** :code:`model.parameters()` returns the Java-weight dictionary (used with
:code:`load_state_dict` / :code:`state_dict`). For the optimizer, always use
:code:`model.tensor_parameters()`.

Example: training an RNN with Adam
----------------------------------

.. code-block:: python

    import torch
    from neuralogic import Model
    from neuralogic.core import Settings
    from neuralogic.dataset import Dataset, Sample
    from neuralogic.nn.module.general import RNN

    # Build a PyNeuraLogic RNN with the torch backend
    model = Model()
    model += RNN(input_size, hidden_size, "h", "f", "h0", arity=0)
    model = model.build(Settings(), torch=True)

    # Get PyTorch-compatible parameters
    params = model.tensor_parameters()
    optimizer = torch.optim.Adam(params, lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # Build the dataset once
    dataset = Dataset([
        Sample(
            R.h(seq_len)[target],
            [R.h0[h0_values], *[R.f(i + 1)[x_i] for i, x_i in enumerate(input_sequence)]],
        ),
    ])
    built_dataset = model.build_dataset(dataset)

    # Standard PyTorch training loop
    for epoch in range(epochs):
        output = model(built_dataset)          # returns a torch.Tensor with autograd
        loss = loss_fn(output[-1], target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


Integration 2: NeuraLogic ``nn.Module`` Wrapper
************************************************

:py:class:`~neuralogic.nn.torch_function.NeuraLogic` wraps a PyNeuraLogic template as a standard
:code:`torch.nn.Module`. You insert it into an :code:`nn.Sequential` pipeline alongside ordinary
PyTorch layers (:code:`nn.Linear`, :code:`nn.Tanh`, etc.) — it takes a tensor, returns a tensor, and
supports autograd.

When to use this
  You have a conventional PyTorch network and want one layer to perform relational reasoning — for
  example, a CNN feeding into a logic layer that enforces structural constraints on the features.

How it works
------------

The :code:`NeuraLogic` module pre-builds your template into a single static computation graph. At each
forward pass, the :code:`to_logic` callback you provide receives the incoming tensor(s) and maps them
to logic facts. The PyNeuraLogic engine runs inference and returns a tensor, which then flows into the
next PyTorch layer. During :code:`backward()`, gradients flow back through the PyNeuraLogic weights.

Constructor
-----------

.. code-block:: python

    NeuraLogic(
        template,           # Model — the PyNeuraLogic template
        input_facts,        # list[BaseRelation | Rule] — initial facts with shapes
        output_relation,    # BaseRelation — the query (output predicate)
        to_logic,           # Callable — maps tensor inputs → list of facts with values
        settings=None,      # Settings | None
        dtype=torch.float32,
    )

Constraints
-----------

**You must provide a ``to_logic`` mapping.** There is no automatic tensor-to-fact conversion. You
write a callback that receives the output of the preceding layer and returns a list of facts with
values assigned. This is the bridge between the tensor world and the logic world.

**One static computation graph.** The graph is built once from the initial facts you provide. At each
forward pass, only the fact *values* change — the graph structure stays the same. This is different
from the usual PyNeuraLogic workflow where each example gets its own graph.

**Weights update during backward, not optimizer step.** The :code:`NeuraLogic` module updates its own
weights inside :code:`backward()`. The outer optimizer's :code:`step()` call does **not** affect
NeuraLogic parameters (it only steps the surrounding PyTorch layers).

**Parameters hidden from ``model.parameters()``.** To keep :code:`nn.Sequential` compatibility, the
module exposes a dummy empty parameter. The real learnable parameters live on
:code:`neuralogic_layer.model.parameters()`. Likewise, use
:code:`neuralogic_layer.model.load_state_dict(...)` to restore weights.

Example: XOR with a NeuraLogic layer
------------------------------------

Start with a pure-PyTorch baseline:

.. code-block:: python

    import torch

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8, bias=False),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1, bias=False),
        torch.nn.Tanh(),
    )

    xs = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=torch.float32)

    ys = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

Define the PyNeuraLogic template and the tensor-to-logic mapping:

.. code-block:: python

    from neuralogic import R, Model
    from neuralogic.core import F

    template = Model()
    template += (R.xor[1, 8] <= R.xy) | [F.identity]
    template += R.xor / 0 | [F.identity]

    def to_logic(tensor_input):
        return [
            R.xy[tensor_input],     # assign the incoming [8] tensor to fact R.xy
        ]

Replace the second :code:`nn.Linear` with the NeuraLogic layer:

.. code-block:: python

    from neuralogic.nn.torch_function import NeuraLogic

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8, bias=False),
        torch.nn.Tanh(),
        NeuraLogic(template, [R.xy[8,]], R.xor, to_logic),
        torch.nn.Tanh(),
    )

Train as usual:

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


Choosing the right integration
*******************************

+---------------------------------------------+-------------------------------------------+
| Scenario                                    | Use                                       |
+=============================================+===========================================+
| PyNeuraLogic *is* your model — you just     | :ref:`Core PyTorch Backend                |
| want Adam/SGD instead of the built-in       | <pytorch-integration-1>`                  |
| trainer.                                    |                                           |
+---------------------------------------------+-------------------------------------------+
| You want a logic layer *inside* a larger    | :ref:`NeuraLogic nn.Module                |
| ``nn.Sequential`` pipeline.                 | <pytorch-integration-2>`                  |
+---------------------------------------------+-------------------------------------------+
| Your input is already a PyNeuraLogic        | :ref:`Core PyTorch Backend                |
| dataset (facts, samples, queries).          | <pytorch-integration-1>`                  |
+---------------------------------------------+-------------------------------------------+
| Your input is a raw tensor and you need     | :ref:`NeuraLogic nn.Module                |
| tensor-in/tensor-out.                       | <pytorch-integration-2>`                  |
+---------------------------------------------+-------------------------------------------+


API Reference
*************

- :py:class:`~neuralogic.core.torch.tensor.NeuralogicOptTensor` — :code:`torch.Tensor` subclass wrapping a Java weight
- :py:class:`~neuralogic.nn.torch_function.NeuraLogic` — :code:`nn.Module` wrapper for embedding a template in a PyTorch pipeline
