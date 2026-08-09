import pytest
import torch

from neuralogic.core import Combination, Model, R, Settings, Transformation, V
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.loss import MSE
from neuralogic.nn.optim import SGD, Adam

LEARNING_RATE = 0.1
INPUT = 3.0
TARGET = 1.0
START = 0.5
STEPS = 5


def _built(optimizer):
    """out = w * input, squared error against a target - one weight, so nothing else can move."""
    model = Model()
    model += R.source("a")[INPUT].fixed()
    model += (R.out(V.X)["w":1, 1] <= R.source(V.X)) | [Combination.SUM, Transformation.IDENTITY]
    built = model.build(
        Settings(
            optimizer=optimizer,
            error_function=MSE(),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    index = next(i for i, name in state["weight_names"].items() if str(name).strip() == "w")
    state["weights"][index] = START
    built.load_state_dict(state)
    return built, index, built.build_dataset(Dataset([Sample(R.out("a")[TARGET], [R.exists("a")])]))


def _neuralogic_trajectory(optimizer):
    built, index, dataset = _built(optimizer)
    trajectory = []
    for _ in range(STEPS):
        built.train(dataset, epochs=1)
        trajectory.append(float(built.state_dict()["weights"][index]))
    return trajectory


def _torch_trajectory(optimizer_class):
    weight = torch.nn.Parameter(torch.tensor([START], dtype=torch.float64))
    optimizer = optimizer_class([weight], lr=LEARNING_RATE)
    trajectory = []
    for _ in range(STEPS):
        optimizer.zero_grad()
        ((weight * INPUT - TARGET) ** 2).sum().backward()
        optimizer.step()
        trajectory.append(float(weight.item()))
    return trajectory


@pytest.mark.parametrize(
    "optimizer, torch_optimizer",
    [(SGD(lr=LEARNING_RATE), torch.optim.SGD), (Adam(lr=LEARNING_RATE), torch.optim.Adam)],
    ids=["sgd", "adam"],
)
def test_optimizer_step_matches_torch(optimizer, torch_optimizer):
    """Stepping the same model on the same data has to move the weight the same way as Torch does.

    Both engines compute their own gradient, so this alone would not say which of the two differed - which
    is what test_gradient_is_the_one_torch_sees below settles.
    """
    for step, (ours, theirs) in enumerate(zip(_neuralogic_trajectory(optimizer), _torch_trajectory(torch_optimizer))):
        assert ours == pytest.approx(theirs, abs=1e-9), f"step {step + 1}"


def test_gradient_is_the_one_torch_sees():
    """The gradients going into the optimizers agree, so an optimizer comparison is about the optimizer.

    Recovered from a plain SGD step, where the update is exactly the learning rate times the gradient - the
    same recovery under Adam would give `lr * sign(g)` instead, which is not a gradient at all.
    """
    built, index, dataset = _built(SGD(lr=LEARNING_RATE))
    before = float(built.state_dict()["weights"][index])
    built.train(dataset, epochs=1)
    ours = (before - float(built.state_dict()["weights"][index])) / LEARNING_RATE

    weight = torch.nn.Parameter(torch.tensor([START], dtype=torch.float64))
    ((weight * INPUT - TARGET) ** 2).sum().backward()

    assert ours == pytest.approx(float(weight.grad.item()), abs=1e-9)
