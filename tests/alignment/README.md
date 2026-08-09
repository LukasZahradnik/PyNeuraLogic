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

## The rule these follow

Every test here was watched failing against a deliberately broken build before being committed, and the
commit message says which break and what it did. A test that has never failed has not been shown to test
anything. Where a test was found *not* to catch something it might be assumed to cover, that is written into
its docstring rather than left implied - see the note on `.parallel()` in `test_batching_alignment.py`.

All of these pass against the jar this branch ships with, so they can be run as they are. One case is
deliberately held back rather than added: a `(4, 1)` weight against a one-element input needs `4511aa59` on
the NeuraLogic `bugfixes-ai` branch, and joins the parametrisation in `test_linear_alignment.py` when the
bundled jar next carries it. The docstring there says so.
