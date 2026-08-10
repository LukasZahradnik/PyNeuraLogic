import pytest
import torch

from neuralogic.core import Combination, Model, R, Settings, Transformation, V
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.loss import MSE, CrossEntropy
from neuralogic.nn.optim import SGD

LEARNING_RATE = 0.1
INPUT = 0.7
START = 0.9
TARGET = 1.0


def _built(transformation, error):
    """out = transformation(w * input), so both the value and its slope come from one weight.

    The transformation is stated on the queried head, which output-function inference leaves alone - so this
    also stands as a check that it still does.
    """
    model = Model()
    model += R.source("a")[INPUT].fixed()
    model += (R.out(V.X)["w":1, 1] <= R.source(V.X)) | [Combination.SUM, Transformation.IDENTITY]
    model += R.out / 1 | [transformation]
    built = model.build(
        Settings(
            optimizer=SGD(lr=LEARNING_RATE),
            error_function=error,
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    index = next(i for i, name in state["weight_names"].items() if str(name).strip() == "w")
    state["weights"][index] = START
    built.load_state_dict(state)
    return built, index, built.build_dataset(Dataset([Sample(R.out("a")[TARGET], [R.exists("a")])]))


def _value_and_gradient(transformation, error):
    built, index, dataset = _built(transformation, error)
    value = float(built(dataset)[0])
    before = float(built.state_dict()["weights"][index])
    built.train(dataset, epochs=1)
    gradient = (before - float(built.state_dict()["weights"][index])) / LEARNING_RATE
    return value, gradient


def _torch_value_and_gradient(activation, loss):
    weight = torch.nn.Parameter(torch.tensor([START], dtype=torch.float64))
    output = activation(weight * INPUT)
    loss(output, torch.tensor([TARGET], dtype=torch.float64)).sum().backward()
    return float(output.item()), float(weight.grad.item())


@pytest.mark.parametrize(
    "transformation, activation",
    [
        (Transformation.IDENTITY, lambda x: x),
        (Transformation.SIGMOID, torch.sigmoid),
        (Transformation.TANH, torch.tanh),
        (Transformation.RELU, torch.relu),
        (Transformation.EXP, torch.exp),
        (Transformation.SQRT, torch.sqrt),
        (Transformation.LOG, torch.log),
        (Transformation.INVERSE, lambda x: 1 / x),
        (Transformation.LEAKY_RELU, lambda x: torch.nn.functional.leaky_relu(x, 0.01)),
        (Transformation.SIGNUM, torch.sign),
    ],
    ids=["identity", "sigmoid", "tanh", "relu", "exp", "sqrt", "log", "inverse", "leaky_relu", "signum"],
)
def test_transformation_matches_torch(transformation, activation):
    """An activation has to agree with Torch in value and in what it does to the gradient passing through."""
    value, gradient = _value_and_gradient(transformation, MSE())
    expected_value, expected_gradient = _torch_value_and_gradient(
        activation, lambda out, target: (out - target) ** 2
    )

    assert value == pytest.approx(expected_value, abs=1e-9)
    assert gradient == pytest.approx(expected_gradient, abs=1e-9)


@pytest.mark.parametrize(
    "error, loss",
    [
        (MSE(), lambda out, target: (out - target) ** 2),
        (
            CrossEntropy(with_logits=False),
            lambda out, target: torch.nn.functional.binary_cross_entropy(out, target),
        ),
    ],
    ids=["squared_diff", "cross_entropy"],
)
def test_error_function_matches_torch(error, loss):
    """The loss has to agree with Torch in the gradient it sends back, which is what training sees of it.

    A squashed output on both sides, so cross-entropy is compared where it is defined - and so the
    with_logits=False convention is pinned rather than assumed.
    """
    _, gradient = _value_and_gradient(Transformation.SIGMOID, error)
    _, expected = _torch_value_and_gradient(torch.sigmoid, loss)

    assert gradient == pytest.approx(expected, abs=1e-9)


def test_the_error_sums_over_a_vector_output_rather_than_averaging():
    """MSE here adds the squared differences up; torch.nn.MSELoss divides by how many there were.

    A scalar output cannot tell the two apart, and neither can a comparison run on Adam - its step follows
    the sign of the gradient far more than the size, so it agrees with either convention to about 1e-7.
    Under plain SGD they are a factor of the output width apart, which is what this pins.
    """
    width = 3
    inputs = [0.7, -0.4, 0.2]
    weight = [[0.5, -0.2, 0.1], [0.3, 0.4, -0.6], [-0.1, 0.2, 0.8]]
    target = [1.0, 0.0, -0.5]

    model = Model()
    model += R.source("a")[inputs].fixed()
    model += (R.out(V.X) <= R.source(V.X)["w":width, width]) | [Combination.SUM, Transformation.IDENTITY]
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
    state["weights"][index] = weight
    built.load_state_dict(state)
    dataset = built.build_dataset(Dataset([Sample(R.out("a")[target], [R.exists("a")])]))

    before = built.state_dict()["weights"][index]
    built.train(dataset, epochs=1)
    after = built.state_dict()["weights"][index]
    gradient = [(b - a) / LEARNING_RATE for rb, ra in zip(before, after) for b, a in zip(rb, ra)]

    summed = _torch_matrix_gradient(inputs, weight, target, lambda diff: diff.sum())
    averaged = _torch_matrix_gradient(inputs, weight, target, lambda diff: diff.mean())

    assert gradient == pytest.approx(summed, abs=1e-9)
    assert max(abs(ours - theirs) for ours, theirs in zip(gradient, averaged)) > 1e-3


def _torch_matrix_gradient(inputs, weight, target, reduce):
    parameter = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float64))
    output = parameter @ torch.tensor(inputs, dtype=torch.float64)
    reduce((output - torch.tensor(target, dtype=torch.float64)) ** 2).backward()
    return [entry for row in parameter.grad.tolist() for entry in row]


VECTOR_INPUT = [0.7, -0.4, 0.2]
VECTOR_WEIGHT = [[0.5, -0.2, 0.1], [0.3, 0.4, -0.6], [-0.1, 0.2, 0.8]]
VECTOR_TARGET = [1.0, 0.0, -0.5]


def _vector_value_and_gradient(transformation, error=None, target=None):
    """out = transformation(W @ x) over a whole vector at once, stated on the queried head."""
    model = Model()
    model += R.source("a")[VECTOR_INPUT].fixed()
    model += (R.out(V.X) <= R.source(V.X)["w":3, 3]) | [Combination.SUM, Transformation.IDENTITY]
    model += R.out / 1 | [transformation]
    built = model.build(
        Settings(
            optimizer=SGD(lr=LEARNING_RATE),
            error_function=error if error is not None else MSE(),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    index = next(i for i, name in state["weight_names"].items() if str(name).strip() == "w")
    state["weights"][index] = VECTOR_WEIGHT
    built.load_state_dict(state)
    dataset = built.build_dataset(
        Dataset([Sample(R.out("a")[target if target is not None else VECTOR_TARGET], [R.exists("a")])])
    )

    value = [float(v) for v in built(dataset)[0]]
    before = built.state_dict()["weights"][index]
    built.train(dataset, epochs=1)
    after = built.state_dict()["weights"][index]
    gradient = [(b - a) / LEARNING_RATE for rb, ra in zip(before, after) for b, a in zip(rb, ra)]
    return value, gradient


def test_softmax_matches_torch_including_its_off_diagonal_gradient():
    """Softmax is the first transformation here whose Jacobian is not diagonal.

    Every activation checked above acts on one number at a time, so a wrong derivative can only be wrong in
    that one place. Softmax mixes the whole vector, and each output depends on every input - so this is the
    one that says the engine carries a full Jacobian back rather than an elementwise slope.
    """
    value, gradient = _vector_value_and_gradient(Transformation.SOFTMAX)

    weight = torch.nn.Parameter(torch.tensor(VECTOR_WEIGHT, dtype=torch.float64))
    output = torch.softmax(weight @ torch.tensor(VECTOR_INPUT, dtype=torch.float64), dim=0)
    ((output - torch.tensor(VECTOR_TARGET, dtype=torch.float64)) ** 2).sum().backward()

    assert value == pytest.approx(output.tolist(), abs=1e-9)
    assert gradient == pytest.approx([e for row in weight.grad.tolist() for e in row], abs=1e-9)


PROBABILITY_TARGET = [0.6, 0.3, 0.1]


def test_cross_entropy_with_logits_matches_torch_on_a_scalar():
    """`with_logits=True` is SOFTENTROPY, which takes the raw logit and squashes inside the loss itself.

    The `with_logits=False` case above is compared behind a sigmoid; this is the other spelling, where
    nothing squashes the head and the loss is expected to. Getting the two the wrong way round produces a
    plausible number rather than an error, which is why both are pinned.
    """
    _, gradient = _value_and_gradient(Transformation.IDENTITY, CrossEntropy(with_logits=True))
    _, expected = _torch_value_and_gradient(
        lambda x: x,
        lambda out, target: torch.nn.functional.binary_cross_entropy_with_logits(out, target),
    )

    assert gradient == pytest.approx(expected, abs=1e-9)


def test_softmax_cross_entropy_over_a_vector_matches_torch():
    """Over a vector, SOFTENTROPY fuses the softmax into the loss - `target - softmax(logit)`.

    The source calls that a nice simplification of doing the two separately, and it is; but a fused form is
    exactly the kind that drifts from the composition it replaces without anything noticing. Compared here
    against torch spelling it out as `-(target * log_softmax(logit)).sum()`.
    """
    _, gradient = _vector_value_and_gradient(
        Transformation.IDENTITY, CrossEntropy(with_logits=True), target=PROBABILITY_TARGET
    )

    weight = torch.nn.Parameter(torch.tensor(VECTOR_WEIGHT, dtype=torch.float64))
    logit = weight @ torch.tensor(VECTOR_INPUT, dtype=torch.float64)
    loss = -(torch.tensor(PROBABILITY_TARGET, dtype=torch.float64) * torch.log_softmax(logit, 0)).sum()
    loss.backward()

    assert gradient == pytest.approx([e for row in weight.grad.tolist() for e in row], abs=1e-9)
