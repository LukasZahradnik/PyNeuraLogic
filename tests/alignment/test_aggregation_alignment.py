import pytest
import torch

from neuralogic.core import Aggregation, Combination, Model, R, Settings, Transformation, V
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.loss import MSE
from neuralogic.nn.optim import SGD

LEARNING_RATE = 0.1
WEIGHT = [[0.5, -0.2], [0.1, 0.4]]
TARGET = [1.0, -0.5]
# Chosen so no two groundings share a maximum or a minimum in either component - a tie would let a wrong
# choice of which input to credit still produce the right answer.
SOURCES = {"a": [0.7, -0.3], "b": [-0.4, 0.9], "c": [0.2, 0.15]}


def _flat(rows):
    return [entry for row in rows for entry in row]


def _neuralogic(aggregation, target=None):
    """One rule grounded once per source, then aggregated: the weight is shared by every grounding."""
    model = Model()
    for name, value in SOURCES.items():
        model += R.source(name)[value].fixed()
    model += (R.weighted(V.X) <= R.source(V.X)["w":2, 2]) | [Combination.SUM, Transformation.IDENTITY]
    model += (R.out("a") <= R.weighted(V.X)) | [aggregation, Transformation.IDENTITY]

    built = model.build(
        Settings(
            optimizer=SGD(lr=LEARNING_RATE),
            error_function=MSE(),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    index = next(i for i, name in state["weight_names"].items() if str(name).strip() == "w")
    state["weights"][index] = WEIGHT
    built.load_state_dict(state)

    facts = [R.exists(name) for name in SOURCES]
    dataset = built.build_dataset(Dataset([Sample(R.out("a")[target if target is not None else TARGET], facts)]))

    produced = built(dataset)[0]
    value = [float(produced)] if isinstance(produced, float) else [float(v) for v in produced]
    before = built.state_dict()["weights"][index]
    built.train(dataset, epochs=1)
    after = built.state_dict()["weights"][index]
    gradient = [(b - a) / LEARNING_RATE for rb, ra in zip(before, after) for b, a in zip(rb, ra)]
    return [float(v) for v in value], gradient


def _torch(pool):
    weight = torch.nn.Parameter(torch.tensor(WEIGHT, dtype=torch.float64))
    stacked = torch.stack([weight @ torch.tensor(v, dtype=torch.float64) for v in SOURCES.values()])
    output = pool(stacked)
    ((output - torch.tensor(TARGET, dtype=torch.float64)) ** 2).sum().backward()
    return output.tolist(), _flat(weight.grad.tolist())


@pytest.mark.parametrize(
    "aggregation, pool",
    [
        (Aggregation.AVG, lambda stacked: stacked.mean(0)),
        (Aggregation.SUM, lambda stacked: stacked.sum(0)),
        (Aggregation.MAX, lambda stacked: stacked[stacked.sum(1).argmax()]),
        (Aggregation.MIN, lambda stacked: stacked[stacked.sum(1).argmin()]),
    ],
    ids=["avg", "sum", "max", "min"],
)
def test_aggregation_matches_torch_pooling(aggregation, pool):
    """How many groundings a rule has is a property of the data, so aggregation has no fixed-shape analogue.

    Pooling over a stack is the nearest thing torch has, and it is the right comparison: one weight is shared
    by every grounding, so the gradient has to come back through all of them. Note what MAX and MIN reduce to
    - a whole grounding picked by the sum of its components, not a componentwise pool.
    """
    value, gradient = _neuralogic(aggregation)
    expected_value, expected_gradient = _torch(pool)

    assert value == pytest.approx(expected_value, abs=1e-9)
    assert gradient == pytest.approx(expected_gradient, abs=1e-9)


def test_max_picks_a_whole_grounding_rather_than_pooling_componentwise():
    """MAX orders groundings by the sum of their components and returns the winner whole.

    torch.max(dim=0) instead takes the largest value in each component separately, which can assemble a
    vector no grounding ever had. The logical reading is defensible - the best grounding is a thing, a
    componentwise mixture of several is not - but it is not what the torch spelling means, so it is asserted
    here rather than left to be discovered.
    """
    value, _ = _neuralogic(Aggregation.MAX)
    stacked = torch.stack(
        [torch.tensor(WEIGHT, dtype=torch.float64) @ torch.tensor(v, dtype=torch.float64) for v in SOURCES.values()]
    )

    assert value == pytest.approx(stacked[stacked.sum(1).argmax()].tolist(), abs=1e-9)
    assert value != pytest.approx(stacked.max(0).values.tolist(), abs=1e-9)


def test_count_reports_how_many_groundings_and_sends_nothing_back():
    """COUNT answers with the number of groundings, which no weight can move.

    So the value is a property of the data rather than of the model, and the gradient through it has to be
    zero - torch's nearest equivalent is `len`, which is not differentiable at all. The backend says as much
    with a warning; what matters is that it also acts on it, since a count that leaked a gradient would train
    a weight towards changing how many times a rule fires.
    """
    # a scalar target, since COUNT answers with one number - a vector one is only found out in the
    # backward pass, as "scalar increment by vector", which is its own recorded complaint
    value, gradient = _neuralogic(Aggregation.COUNT, target=3.0)

    assert value == pytest.approx([float(len(SOURCES))] * len(value), abs=1e-9)
    assert gradient == pytest.approx([0.0] * len(gradient), abs=1e-12)
