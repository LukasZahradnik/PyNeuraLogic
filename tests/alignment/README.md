# Alignment tests

What a classical deep learning model written here computes, checked against torch computing the same thing.
Small, fixed, fast - the whole folder runs in about six seconds.

They are layered, each one leaning on the one below:

| file | what it compares |
| --- | --- |
| `test_gradient_correctness.py` | the backward pass against a central difference of the model's own forward pass - no torch, so it needs no shared convention |
| `test_primitive_alignment.py` | six transformations and two error functions, value and derivative both |
| `test_linear_alignment.py` | matrix weights against `torch.nn.Linear(bias=False)`, including a stacked layer, which is what exercises the transposition on the way down |
| `test_aggregation_alignment.py` | AVG, SUM, MAX, MIN against torch pooling |
| `test_optimizer_alignment.py` | SGD and Adam trajectories over five steps |
| `test_recurrent_alignment.py` | RNN, LSTM and GRU on one forward and one SGD step, compared on the weights |
| `test_batching_alignment.py` | a batch step against torch on the same groups, plus epochs and repeatability |

## Two things to keep in mind when adding to these

**Compare on SGD, not Adam.** Adam's step follows the sign of the gradient far more than its size, so a
comparison run on it cannot see a scale error. `test_recurrent_modules.py` in the parent folder compares on
Adam for 500 epochs at `atol=1e-4`; with the loss gradient halved it still passes, while
`test_recurrent_alignment.py` fails on every weight. Under plain SGD the step is exactly the learning rate
times the gradient, which is also the only condition under which recovering a gradient from a weight step is
valid at all.

**The error function sums where torch averages.** `MSE` here adds the squared differences up;
`torch.nn.MSELoss` divides by how many there were. A scalar output cannot tell the two apart - use a vector
one. The same holds for a batch: the update is the sum over its samples, so the size of a batch changes the
size of the step.

## Where the activation coverage stops, on purpose

Ten transformations are compared, and the two whose Jacobian is not diagonal - softmax and layer norm - are
the ones that would catch an engine carrying an elementwise slope where a full one belongs. The rest are
left alone deliberately rather than forgotten: `sparsemax` has no torch counterpart to compare against
without writing the reference too, `lukasiewicz` has none at all, and `concat`, `slice`, `reshape`, `transp`
and `reverse` rearrange values rather than compute with them, so what could go wrong in them is shape
handling and not arithmetic. Same for the `softmax` and `concat` aggregations. Anyone wanting them should
know they are a deliberate gap, not an oversight.

## Check the model has no weights you are not setting

A comparison only means anything if every weight is one you put there. A hand-written template picks up
implicit weights on rules you did not think of as weighted, and those stay randomly initialised - which
shows up as a comparison that fails *and* gives different numbers on each run. If a probe is not
reproducible, count the weights before doubting the engine:

    state = built.state_dict()
    assert len(state["weight_names"]) == 1, state["weight_names"]

That cost an afternoon on the GNN modules, where the real modules turned out to have exactly the weights
being set and only the hand-written reconstruction did not.

## The rule these follow

Every test here was watched failing against a deliberately broken build before being committed, and the
commit message says which break and what it did. A test that has never failed has not been shown to test
anything. Where a test was found *not* to catch something it might be assumed to cover, that is written into
its docstring rather than left implied - see the note on `.parallel()` in `test_batching_alignment.py`.

All of these pass against the jar this branch ships with, so they can be run as they are. One case is
deliberately held back rather than added: a `(4, 1)` weight against a one-element input needs `4511aa59` on
the NeuraLogic `bugfixes-ai` branch, and joins the parametrisation in `test_linear_alignment.py` when the
bundled jar next carries it. The docstring there says so.
