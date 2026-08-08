import pytest

from neuralogic.core import Aggregation, Combination, Model, R, Settings, Transformation, V
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.loss import MSE, CrossEntropy
from neuralogic.nn.optim import SGD

LEARNING_RATE = 0.01
STEP = 1e-5
VECTOR, OTHER = [3.0, -1.0], [1.0, 2.0]
SCALAR = 3.0
TARGET = 1.0


def _single():
    model = Model()
    model += R.source("a")[VECTOR].fixed()
    model += (R.weighted(V.X)["w":1, 2] <= R.source(V.X)) | [Transformation.IDENTITY]
    model += (R.out(V.X) <= R.weighted(V.X)) | [Combination.SUM, Transformation.IDENTITY]
    return model, [[0.5, 0.25]], [R.exists("a")], MSE()


def _shared():
    """One weight reaching the output through two rules that merge."""
    model = Model()
    model += R.source("a")[VECTOR].fixed()
    model += (R.weighted(V.X)["w":1, 2] <= R.source(V.X)) | [Transformation.IDENTITY]
    model += (R.left(V.X) <= R.weighted(V.X)) | [Transformation.IDENTITY]
    model += (R.right(V.X) <= R.weighted(V.X)) | [Transformation.IDENTITY]
    model += (R.out(V.X) <= (R.left(V.X), R.right(V.X))) | [Combination.SUM, Transformation.IDENTITY]
    return model, [[0.5, 0.25]], [R.exists("a")], MSE()


def _averaged():
    """Two groundings of one rule averaged, so the gradient carries the mean of the inputs."""
    model = Model()
    model += R.source("a")[VECTOR].fixed()
    model += R.source("b")[OTHER].fixed()
    model += (R.weighted(V.X)["w":1, 2] <= R.source(V.X)) | [Transformation.IDENTITY]
    model += (R.out("a") <= R.weighted(V.X)) | [Aggregation.AVG, Transformation.IDENTITY]
    return model, [[0.5, 0.25]], [R.exists("a")], MSE()


def _element_product():
    """Two weighted atoms multiplied, so each one's gradient carries the other's value."""
    model = Model()
    model += R.source("a")[VECTOR].fixed()
    model += R.source("b")[OTHER].fixed()
    model += (R.weighted(V.X)["w":1, 2] <= R.source(V.X)) | [Transformation.IDENTITY]
    model += (R.out("a") <= (R.weighted("a"), R.weighted("b"))) | [
        Combination.ELPRODUCT,
        Transformation.IDENTITY,
    ]
    return model, [[0.5, 0.25]], [R.exists("a")], MSE()


def _cross_entropy():
    """A squashed output under cross-entropy, where the engine rewrites the output activation itself."""
    model, weight, facts, _ = _single()
    model += R.out / 1 | [Transformation.SIGMOID]
    return model, weight, facts, CrossEntropy(with_logits=False)


def _recurrent():
    """One rule grounded once per timestep through the recurrent edge: out = w**3 * x."""
    model = Model()
    model += R.h(0)[SCALAR].fixed()
    model += (R.h(V.T)["w":1, 1] <= (R.h(V.Z), R.special.next(V.Z, V.T))) | [Transformation.IDENTITY]
    model += (R.out("a") <= R.h(3)) | [Transformation.IDENTITY]
    facts = [R.h(0)[SCALAR]] + [R.special.next(t, t + 1) for t in range(3)]
    return model, [[0.5]], facts, MSE()


def _reused():
    """One weight occurring twice in the same graph at different depths: out = w*x + w**2*x."""
    model = Model()
    model += R.source("a")[SCALAR].fixed()
    model += (R.once(V.Z)["w":1, 1] <= R.source(V.Z)) | [Transformation.IDENTITY]
    model += (R.twice(V.Z)["w":1, 1] <= R.once(V.Z)) | [Transformation.IDENTITY]
    model += (R.out(V.Z) <= (R.once(V.Z), R.twice(V.Z))) | [Combination.SUM, Transformation.IDENTITY]
    return model, [[0.5]], [R.exists("a")], MSE()


def _build(topology, weight):
    model, _, facts, error = topology()
    built = model.build(
        Settings(
            optimizer=SGD(lr=LEARNING_RATE),
            error_function=error,
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    names = [index for index, name in state["weight_names"].items() if str(name).strip() == "w"]
    assert len(state["weights"]) == 1, "the comparison needs exactly one learnable weight"
    state["weights"][names[0]] = weight if len(weight[0]) > 1 else weight[0][0]
    built.load_state_dict(state)
    return built, names[0], built.build_dataset(Dataset([Sample(R.out("a")[TARGET], facts)]))


def _flat(value):
    if isinstance(value, (int, float)):
        return [float(value)]
    return [entry for row in value for entry in row] if isinstance(value[0], list) else list(value)


def _loss(topology, weight):
    built, _, dataset = _build(topology, weight)
    results = built.validate(dataset)
    return float(sum(float(error) for _, _, error in results))


def _analytic_gradient(topology, weight):
    """The step is w -= lr * gradient under plain SGD, so the gradient is what the step undoes."""
    built, index, dataset = _build(topology, weight)
    before = _flat(built.state_dict()["weights"][index])
    built.train(dataset, epochs=1)
    after = _flat(built.state_dict()["weights"][index])
    return [(b - a) / LEARNING_RATE for b, a in zip(before, after)]


def _numeric_gradient(topology, weight):
    gradient = []
    for position in range(len(_flat(weight))):
        shifted = []
        for sign in (+1, -1):
            moved = [list(row) for row in weight]
            row, column = divmod(position, len(weight[0]))
            moved[row][column] += sign * STEP
            shifted.append(_loss(topology, moved))
        gradient.append((shifted[0] - shifted[1]) / (2 * STEP))
    return gradient


@pytest.mark.parametrize(
    "topology",
    [_single, _shared, _averaged, _element_product, _cross_entropy, _recurrent, _reused],
    ids=["single", "shared", "averaged", "element_product", "cross_entropy", "recurrent", "reused"],
)
def test_gradient_matches_finite_differences(topology):
    """The analytic gradient must agree with a central difference of the model's own forward pass.

    Nothing else is needed to catch a wrong backward: if an engine's gradient disagrees with the slope of
    its own loss, the gradient is wrong, whatever the conventions.
    """
    _, weight, _, _ = topology()
    analytic = _analytic_gradient(topology, weight)
    numeric = _numeric_gradient(topology, weight)

    for expected, actual in zip(numeric, analytic):
        assert actual == pytest.approx(expected, abs=1e-6)
