import numpy as np
import pytest
import torch

from neuralogic.core import Template, Settings, R
from neuralogic.dataset import Dataset
from neuralogic.nn.loss import MSE

from neuralogic.nn.module import GRU, RNN, LSTM
from neuralogic.optim import Adam


def test_gru_module_forward():
    """Test that PyNeuraLogic GRU layer computes the same as pytorch GRU layer"""
    in_size = 1
    steps = 1
    hid_size = 1
    layers = 1

    rnn = torch.nn.GRU(in_size, hid_size, layers, bias=False)
    torch_input = torch.randn(steps, in_size)

    h0 = torch.tensor([[1.2023]])
    output, _ = rnn(torch_input, h0)

    template = Template()
    template += GRU(in_size, hid_size, "h", "f", "h0", arity=0)

    model = template.build(Settings(chain_pruning=False, iso_value_compression=False))

    parameters = model.parameters()
    torch_parameters = [parameter.tolist() for parameter in rnn.parameters()]

    parameters["weights"][0] = torch_parameters[0][0]
    parameters["weights"][2] = torch_parameters[0][1]
    parameters["weights"][5] = torch_parameters[0][2]

    parameters["weights"][1] = torch_parameters[1][0]
    parameters["weights"][3] = torch_parameters[1][1]
    parameters["weights"][4] = torch_parameters[1][2]

    model.load_state_dict(parameters)

    dataset = Dataset(
        [[R.f(1)[float(torch_input[0][0])], R.h0[float(h0[0][0])]]],
        [R.h(1)[[1]]],
    )

    bd = model.build_dataset(dataset)

    result = model(bd.samples, train=False)
    assert round(float(output[0][0]), 3) == round(result[0][0], 3)


def test_rnn_module_forward():
    """Test that PyNeuraLogic RNN layer computes the same as pytorch RNN layer"""
    in_size = 1
    hid_size = 1
    steps = 1
    layers = 1

    rnn = torch.nn.RNN(in_size, hid_size, layers, bias=False)
    torch_input = torch.randn(steps, in_size)

    h0 = torch.tensor([[1.2023]])
    output, _ = rnn(torch_input, h0)

    template = Template()
    template += RNN(in_size, hid_size, "h", "f", "h0", arity=0)

    model = template.build(Settings(chain_pruning=False, iso_value_compression=False))

    parameters = model.parameters()
    torch_parameters = [parameter.tolist() for parameter in rnn.parameters()]

    parameters["weights"][0] = torch_parameters[0]
    parameters["weights"][1] = torch_parameters[1]

    model.load_state_dict(parameters)

    dataset = Dataset(
        [[R.f(1)[float(torch_input[0][0])], R.h0[float(h0[0][0])]]],
        [R.h(1)[[1]]],
    )

    bd = model.build_dataset(dataset)

    result = model(bd.samples, train=False)
    assert round(float(output[0][0]), 3) == round(result[0][0], 3)


def test_lstm_module_forward():
    """Test that PyNeuraLogic LSTM layer computes the same as pytorch LSTM layer"""
    in_size = 1
    hid_size = 1
    steps = 1
    layers = 1

    rnn = torch.nn.LSTM(in_size, hid_size, layers, bias=False)
    torch_input = torch.randn(steps, in_size)

    h0 = torch.tensor([[1.2023]])
    c0 = torch.tensor([[1.0123]])
    output, _ = rnn(torch_input, (h0, c0))

    template = Template()
    template += LSTM(in_size, hid_size, "h", "f", "h0", "c0", arity=0)

    model = template.build(Settings(chain_pruning=False, iso_value_compression=False))

    parameters = model.parameters()
    torch_parameters = [parameter.tolist() for parameter in rnn.parameters()]

    parameters["weights"][0] = torch_parameters[0][0]
    parameters["weights"][2] = torch_parameters[0][1]
    parameters["weights"][4] = torch_parameters[0][3]
    parameters["weights"][6] = torch_parameters[0][2]

    parameters["weights"][1] = torch_parameters[1][0]
    parameters["weights"][3] = torch_parameters[1][1]
    parameters["weights"][5] = torch_parameters[1][3]
    parameters["weights"][7] = torch_parameters[1][2]

    model.load_state_dict(parameters)

    dataset = Dataset(
        [[R.f(1)[float(torch_input[0][0])], R.h0[float(h0[0][0])], R.c0[float(c0[0][0])]]],
        [R.h(1)[[1]]],
    )

    bd = model.build_dataset(dataset)

    result = model(bd.samples, train=False)
    assert round(float(output[0][0]), 3) == round(result[0][0], 3)


@pytest.mark.parametrize(
    "error_fun, torch_error_fun, target, epochs",
    [
        (MSE(), torch.nn.MSELoss, [1, 0.8, 0.5], 300),
    ],
)
def test_rnn_module_bacprop(error_fun, torch_error_fun, target, epochs):
    """Test that PyNeuraLogic RNN layer computes the same as pytorch RNN layer (with backprop)"""
    in_size = 3
    hid_size = 3
    steps = 3
    layers = 1

    torch_input = torch.randn((steps, in_size))
    h0 = torch.randn((1, hid_size))

    rnn = torch.nn.RNN(in_size, hid_size, layers, bias=False)

    template = Template()
    template += RNN(in_size, hid_size, "h", "f", "h0", arity=0)

    model = template.build(
        Settings(chain_pruning=False, iso_value_compression=False, optimizer=Adam(lr=0.001), error_function=error_fun)
    )

    parameters = model.parameters()
    torch_parameters = [parameter.tolist() for parameter in rnn.parameters()]

    parameters["weights"][0] = torch_parameters[0]
    parameters["weights"][1] = torch_parameters[1]

    model.load_state_dict(parameters)

    dataset = Dataset(
        [[R.h0[[float(h) for h in h0[0]]], *[R.f(i + 1)[[float(h) for h in torch_input[i]]] for i in range(steps)]]],
        [R.h(steps)[target]],
    )

    bd = model.build_dataset(dataset)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    loss_fun = torch_error_fun()

    for _ in range(epochs):
        output, _ = rnn(torch_input, h0)
        loss = loss_fun(output[-1], torch.tensor(target))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        result, _ = model(bd.samples)
        assert np.allclose([float(x) for x in output[-1]], [float(x) for x in result[0][1]], atol=10e-5)
