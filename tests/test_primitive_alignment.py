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
    ],
    ids=["identity", "sigmoid", "tanh", "relu", "exp", "sqrt"],
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
