import pytest
import torch

from neuralogic.core import Combination, Model, R, Settings, Transformation, V
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.loss import MSE
from neuralogic.nn.optim import SGD

LEARNING_RATE = 0.1


def _spread(count, start):
    """Distinct asymmetric values, so a transposed or reordered matrix cannot pass unnoticed."""
    return [round(start + 0.37 * i, 4) for i in range(count)]


def _matrix(rows, columns):
    return [_spread(columns, 0.5 - 0.23 * row) for row in range(rows)]


def _flat(value):
    if isinstance(value, (int, float)):
        return [float(value)]
    return [entry for row in value for entry in _flat(row)]


def _build(model, weights, target, facts=(R.exists("a"),)):
    built = model.build(
        Settings(
            optimizer=SGD(lr=LEARNING_RATE),
            error_function=MSE(),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    names = {str(name).strip(): index for index, name in state["weight_names"].items()}
    for name, value in weights.items():
        state["weights"][names[name]] = value
    built.load_state_dict(state)
    return built, names, built.build_dataset(Dataset([Sample(R.out("a")[target], list(facts))]))


def _forward_and_gradients(built, names, dataset):
    """The step is w -= lr * gradient under plain SGD, so the gradient is what the step undoes."""
    value = _flat(built(dataset)[0])
    before = {name: _flat(built.state_dict()["weights"][i]) for name, i in names.items()}
    built.train(dataset, epochs=1)
    after = {name: _flat(built.state_dict()["weights"][i]) for name, i in names.items()}
    return value, {
        name: [(b - a) / LEARNING_RATE for b, a in zip(before[name], after[name])] for name in names
    }


def _tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def _squared_error(output, target):
    return ((output - _tensor(target)) ** 2).sum()


@pytest.mark.parametrize("rows, columns", [(3, 2), (2, 3), (1, 4)], ids=["3x2", "2x3", "1x4"])
def test_linear_layer_matches_torch(rows, columns):
    """out = W @ x against torch.nn.Linear(bias=False), on rectangular shapes both ways round.

    A weight on a body atom multiplies that atom, which is the matrix-vector product a linear layer is. The
    shapes are deliberately not square: a transposed matrix would not even have a defined product here.
    """
    inputs, weight, target = _spread(columns, 0.7), _matrix(rows, columns), _spread(rows, -0.4)

    model = Model()
    model += R.source("a")[inputs].fixed()
    model += (R.out(V.X) <= R.source(V.X)["w":rows, columns]) | [Combination.SUM, Transformation.IDENTITY]
    value, gradients = _forward_and_gradients(*_build(model, {"w": weight}, target))

    layer = torch.nn.Linear(columns, rows, bias=False).double()
    with torch.no_grad():
        layer.weight.copy_(_tensor(weight))
    output = layer(_tensor(inputs))
    _squared_error(output, target).backward()

    assert value == pytest.approx(_flat(output.tolist()), abs=1e-9)
    assert gradients["w"] == pytest.approx(_flat(layer.weight.grad.tolist()), abs=1e-9)


def test_projecting_from_one_dimension_takes_a_scalar_input():
    """torch.nn.Linear(1, 4) reads a one-element vector; here the same layer reads a scalar.

    A weight declared (4, 1) arrives as a plain vector of four - the trailing dimension is not kept - so a
    one-element input has to be a scalar rather than a list of one. Written as [x] it raises
    "Incompatible dimensions ... (try transposition)", which is not advice that applies.
    """
    weight, target = _matrix(4, 1), _spread(4, -0.4)

    model = Model()
    model += R.source("a")[0.7].fixed()
    model += (R.out(V.X) <= R.source(V.X)["w":4, 1]) | [Combination.SUM, Transformation.IDENTITY]
    value, gradients = _forward_and_gradients(*_build(model, {"w": weight}, target))

    layer = torch.nn.Linear(1, 4, bias=False).double()
    with torch.no_grad():
        layer.weight.copy_(_tensor(weight))
    output = layer(_tensor([0.7]))
    _squared_error(output, target).backward()

    assert value == pytest.approx(_flat(output.tolist()), abs=1e-9)
    assert gradients["w"] == pytest.approx(_flat(layer.weight.grad.tolist()), abs=1e-9)


def test_stacked_layers_match_torch():
    """W2 @ tanh(W1 @ x): the first layer's gradient is only right if W2 is transposed on the way down.

    A single layer never asks the backward pass to move a gradient through a matrix - it only has to land in
    one. Two layers do, which is where an orientation mistake would show.
    """
    inputs, first, second = _spread(2, 0.7), _matrix(3, 2), _matrix(2, 3)
    target = _spread(2, -0.4)

    model = Model()
    model += R.source("a")[inputs].fixed()
    model += (R.mid(V.X) <= R.source(V.X)["w1":3, 2]) | [Combination.SUM, Transformation.TANH]
    model += (R.out(V.X) <= R.mid(V.X)["w2":2, 3]) | [Combination.SUM, Transformation.IDENTITY]
    value, gradients = _forward_and_gradients(*_build(model, {"w1": first, "w2": second}, target))

    w1, w2 = torch.nn.Parameter(_tensor(first)), torch.nn.Parameter(_tensor(second))
    output = w2 @ torch.tanh(w1 @ _tensor(inputs))
    _squared_error(output, target).backward()

    assert value == pytest.approx(_flat(output.tolist()), abs=1e-9)
    assert gradients["w1"] == pytest.approx(_flat(w1.grad.tolist()), abs=1e-9)
    assert gradients["w2"] == pytest.approx(_flat(w2.grad.tolist()), abs=1e-9)


def test_two_weighted_inputs_sum_like_torch():
    """Wx @ x + Wh @ h in one body, which is the shape of a recurrent cell before it is unrolled."""
    inputs, state = _spread(2, 0.7), _spread(3, -0.2)
    weight_in, weight_state, target = _matrix(3, 2), _matrix(3, 3), _spread(3, -0.4)

    model = Model()
    model += R.source("a")[inputs].fixed()
    model += R.state("a")[state].fixed()
    model += (R.out(V.X) <= (R.source(V.X)["wx":3, 2], R.state(V.X)["wh":3, 3])) | [
        Combination.SUM,
        Transformation.TANH,
    ]
    value, gradients = _forward_and_gradients(
        *_build(model, {"wx": weight_in, "wh": weight_state}, target)
    )

    wx, wh = torch.nn.Parameter(_tensor(weight_in)), torch.nn.Parameter(_tensor(weight_state))
    output = torch.tanh(wx @ _tensor(inputs) + wh @ _tensor(state))
    _squared_error(output, target).backward()

    assert value == pytest.approx(_flat(output.tolist()), abs=1e-9)
    assert gradients["wx"] == pytest.approx(_flat(wx.grad.tolist()), abs=1e-9)
    assert gradients["wh"] == pytest.approx(_flat(wh.grad.tolist()), abs=1e-9)


def test_weight_placement_decides_which_side_of_the_activation_it_is_on():
    """A body weight gives activation(W @ x); a head weight gives W @ activation(x). Both, stated.

    A classical layer is the first one, so this is the choice a reader coming from Torch has to make
    knowingly - the two rules look almost identical and compute different things.
    """
    inputs, weight, target = _spread(2, 0.7), _matrix(3, 2), _spread(3, -0.4)

    body = Model()
    body += R.source("a")[inputs].fixed()
    body += (R.out(V.X) <= R.source(V.X)["w":3, 2]) | [Combination.SUM, Transformation.TANH]

    head = Model()
    head += R.source("a")[inputs].fixed()
    head += (R.out(V.X)["w":3, 2] <= R.source(V.X)) | [Combination.SUM, Transformation.TANH]

    inside, _ = _forward_and_gradients(*_build(body, {"w": weight}, target))
    outside, _ = _forward_and_gradients(*_build(head, {"w": weight}, target))

    assert inside == pytest.approx(torch.tanh(_tensor(weight) @ _tensor(inputs)).tolist(), abs=1e-9)
    assert outside == pytest.approx((_tensor(weight) @ torch.tanh(_tensor(inputs))).tolist(), abs=1e-9)
