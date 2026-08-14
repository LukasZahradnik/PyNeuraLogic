import pytest
import torch

from neuralogic.core import Combination, Model, R, Settings, Transformation, V
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.loss import MSE
from neuralogic.nn.optim import SGD

LEARNING_RATE = 0.1
START = [[0.5, -0.2], [0.1, 0.4]]
NAMES = ["a", "b", "c", "d", "e"]
INPUTS = [[0.7, -0.3], [0.2, 0.9], [-0.5, 0.4], [0.8, 0.1], [-0.2, -0.6]]
TARGETS = [[1.0, 0.0], [0.3, -0.4], [-0.5, 0.8], [0.2, 0.6], [0.9, -0.1]]


def _flat(rows):
    return [entry for row in rows for entry in row]


def _trained(count, batch_size, epochs):
    """One weight shared by every sample, so a batch update is the only thing that can move it."""
    model = Model()
    model += (R.out(V.X) <= R.source(V.X)["w":2, 2]) | [Combination.SUM, Transformation.IDENTITY]
    built = model.build(
        Settings(
            optimizer=SGD(lr=LEARNING_RATE),
            error_function=MSE(reduction="sum"),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    index = next(i for i, name in state["weight_names"].items() if str(name).strip() == "w")
    state["weights"][index] = START
    built.load_state_dict(state)

    samples = [Sample(R.out(NAMES[i])[TARGETS[i]], [R.source(NAMES[i])[INPUTS[i]]]) for i in range(count)]
    built.train(built.build_dataset(Dataset(samples), batch_size=batch_size), epochs=epochs)
    return _flat(built.state_dict()["weights"][index])


def _torch_grouped(groups, epochs=1):
    weight = torch.nn.Parameter(torch.tensor(START, dtype=torch.float64))
    optimizer = torch.optim.SGD([weight], lr=LEARNING_RATE)
    for _ in range(epochs):
        for group in groups:
            optimizer.zero_grad()
            total = sum(
                (
                    (weight @ torch.tensor(INPUTS[i], dtype=torch.float64))
                    - torch.tensor(TARGETS[i], dtype=torch.float64)
                )
                ** 2
                for i in group
            ).sum()
            total.backward()
            optimizer.step()
    return _flat(weight.detach().tolist())


@pytest.mark.parametrize(
    "batch_size, groups",
    [(1, [[0], [1], [2], [3], [4]]), (2, [[0, 1], [2, 3], [4]]), (5, [[0, 1, 2, 3, 4]])],
    ids=["one_at_a_time", "ragged_last_batch", "whole_set_at_once"],
)
def test_batches_step_like_torch_on_the_same_groups(batch_size, groups):
    """A batch update has to be the sum of its samples', which torch gives by summing the batch's losses.

    Note it is a sum and not a mean, matching the error function - so the size of a batch changes the size of
    the step, where torch users would normally expect it not to. Five samples over batches of two also puts
    a short batch at the end, and only agrees if the tail is handled and the samples keep their order.
    """
    assert _trained(5, batch_size, 1) == pytest.approx(_torch_grouped(groups), abs=1e-9)


def test_repeated_epochs_stay_exact():
    """Anything left behind in the update accumulator between passes would show up as drift by epoch three."""
    assert _trained(4, 4, 3) == pytest.approx(_torch_grouped([[0, 1, 2, 3]], epochs=3), abs=1e-9)


def test_the_same_batched_run_twice_gives_the_same_weights():
    """Batch two once accumulated differently between identical seeded processes - gradient cosine 0.896.

    That came from samples in a batch sharing neuron state while being trained on several threads. Exact
    repeatability is the property that broke, so it is the one asserted, rather than closeness.

    Note what this does not cover: putting `.parallel()` back on the minibatch stream leaves every test in
    this file green, on these topologies and on one built around a shared template rule - measured. The
    reproducer suite still catches it, so that is what guards it, not this.
    """
    runs = [_trained(5, 2, 2) for _ in range(3)]
    assert runs[1] == runs[0]
    assert runs[2] == runs[0]
