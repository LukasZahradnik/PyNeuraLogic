import pytest
import torch

from neuralogic.core import Model, R, Settings
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.loss import MSE
from neuralogic.nn.module import GRU, LSTM, RNN
from neuralogic.nn.optim import SGD

LEARNING_RATE = 0.1
INPUT_SIZE, HIDDEN_SIZE, LENGTH = 3, 2, 4
SEED = 7

# (weight of the module, which torch matrix it comes from, which gate block of that matrix).
# Torch stacks the gates row-wise into one matrix per direction, in the order i,f,g,o for LSTM and r,z,n for
# GRU; the modules here keep one weight per gate, in an order of their own.
LAYOUTS = {
    "rnn": [(0, 0, 0), (1, 1, 0)],
    "lstm": [(0, 0, 0), (2, 0, 1), (6, 0, 2), (4, 0, 3), (1, 1, 0), (3, 1, 1), (7, 1, 2), (5, 1, 3)],
    "gru": [(0, 0, 0), (2, 0, 1), (5, 0, 2), (1, 1, 0), (3, 1, 1), (4, 1, 2)],
}


def _flat(value):
    if isinstance(value, (int, float)):
        return [float(value)]
    return [entry for row in value for entry in _flat(row)]


def _torch_module(kind):
    return {"rnn": torch.nn.RNN, "lstm": torch.nn.LSTM, "gru": torch.nn.GRU}[kind](
        INPUT_SIZE, HIDDEN_SIZE, 1, bias=False
    ).double()


def _module(kind):
    if kind == "lstm":
        return LSTM(INPUT_SIZE, HIDDEN_SIZE, "h", "f", "h0", "c0", arity=0)
    return (RNN if kind == "rnn" else GRU)(INPUT_SIZE, HIDDEN_SIZE, "h", "f", "h0", arity=0)


def _block(matrix, index):
    return matrix.detach()[index * HIDDEN_SIZE : (index + 1) * HIDDEN_SIZE].tolist()


def _sample(kind, inputs, h0, c0, target):
    facts = [R.h0[[float(v) for v in h0[0]]]]
    if kind == "lstm":
        facts.insert(0, R.c0[[float(v) for v in c0[0]]])
    facts += [R.f(step + 1)[[float(v) for v in inputs[step]]] for step in range(LENGTH)]
    return Sample(R.h(LENGTH)[target.tolist()], facts)


@pytest.mark.parametrize("kind", ["rnn", "lstm", "gru"])
def test_recurrent_module_matches_torch(kind):
    """One forward and one plain SGD step against the torch module, compared on the weights themselves.

    test_recurrent_modules.py already runs these three against torch for 500 epochs, but on Adam and on the
    output alone. Adam's step is dominated by the sign of the gradient rather than its size, so that
    comparison holds to 1e-7 even against a loss scaled by the width of the output - measured. One SGD step
    is what makes the size of the gradient observable, and the weights are where it shows.
    """
    torch.manual_seed(SEED)
    inputs = torch.randn((LENGTH, INPUT_SIZE), dtype=torch.float64)
    h0 = torch.randn((1, HIDDEN_SIZE), dtype=torch.float64)
    c0 = torch.randn((1, HIDDEN_SIZE), dtype=torch.float64)
    target = torch.randn((HIDDEN_SIZE,), dtype=torch.float64)

    reference = _torch_module(kind)
    matrices = [reference.weight_ih_l0, reference.weight_hh_l0]

    model = Model()
    model += _module(kind)
    built = model.build(
        Settings(
            chain_pruning=False,
            iso_value_compression=False,
            optimizer=SGD(lr=LEARNING_RATE),
            error_function=MSE(reduction="sum"),
        )
    )
    parameters = built.parameters()
    for weight, matrix, block in LAYOUTS[kind]:
        parameters["weights"][weight] = _block(matrices[matrix], block)
    built.load_state_dict(parameters)

    dataset = built.build_dataset(Dataset([_sample(kind, inputs, h0, c0, target)]))
    output, _ = reference(inputs, (h0, c0) if kind == "lstm" else h0)

    assert _flat(built(dataset)[0]) == pytest.approx(_flat(output[-1].tolist()), abs=1e-9)

    optimizer = torch.optim.SGD(reference.parameters(), lr=LEARNING_RATE)
    ((output[-1] - target) ** 2).sum().backward()
    optimizer.step()
    built.train(dataset, epochs=1)

    stepped = built.parameters()["weights"]
    for weight, matrix, block in LAYOUTS[kind]:
        assert _flat(stepped[weight]) == pytest.approx(
            _flat(_block(matrices[matrix], block)), abs=1e-9
        ), f"weight {weight}"
