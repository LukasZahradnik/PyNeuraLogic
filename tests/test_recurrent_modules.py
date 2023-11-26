import numpy as np
import pytest
import torch

from neuralogic.core import Template, Settings, R
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.loss import MSE

from neuralogic.nn.module import GRU, RNN, LSTM
from neuralogic.optim import Adam


@pytest.mark.parametrize(
    "input_size, hidden_size, sequence_len, epochs",
    [
        (10, 5, 10, 500),
    ],
)
def test_gru_module(input_size, hidden_size, sequence_len, epochs):
    """Test that PyNeuraLogic GRU layer computes the same as PyTorch GRU layer (with backprop)"""
    torch_input = torch.randn((sequence_len, input_size))
    h0 = torch.randn((1, hidden_size))
    target = torch.randn((hidden_size,))

    rnn = torch.nn.GRU(input_size, hidden_size, 1, bias=False)

    template = Template()
    template += GRU(input_size, hidden_size, "h", "f", "h0", arity=0)

    model = template.build(
        Settings(chain_pruning=False, iso_value_compression=False, optimizer=Adam(lr=0.001), error_function=MSE())
    )

    parameters = model.parameters()
    torch_parameters = [parameter.tolist() for parameter in rnn.parameters()]

    parameters["weights"][0] = [torch_parameters[0][i] for i in range(0, hidden_size)]
    parameters["weights"][2] = [torch_parameters[0][i] for i in range(1 * hidden_size, 1 * hidden_size + hidden_size)]
    parameters["weights"][5] = [torch_parameters[0][i] for i in range(2 * hidden_size, 2 * hidden_size + hidden_size)]

    parameters["weights"][1] = [torch_parameters[1][i] for i in range(0, hidden_size)]
    parameters["weights"][3] = [torch_parameters[1][i] for i in range(1 * hidden_size, 1 * hidden_size + hidden_size)]
    parameters["weights"][4] = [torch_parameters[1][i] for i in range(2 * hidden_size, 2 * hidden_size + hidden_size)]

    model.load_state_dict(parameters)

    dataset = Dataset(
        [
            Sample(
                R.h(sequence_len)[target.detach().numpy().tolist()],
                [
                    R.h0[[float(h) for h in h0[0]]],
                    *[R.f(i + 1)[[float(h) for h in torch_input[i]]] for i in range(sequence_len)],
                ],
            )
        ]
    )

    bd = model.build_dataset(dataset)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    loss_fun = torch.nn.MSELoss()

    for _ in range(epochs):
        output, _ = rnn(torch_input, h0)
        loss = loss_fun(output[-1], target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        result, _ = model(bd.samples)
        assert np.allclose([float(x) for x in output[-1]], [float(x) for x in result[0][1]], atol=10e-5)


@pytest.mark.parametrize(
    "input_size, hidden_size, sequence_len, epochs",
    [
        (10, 5, 10, 500),
    ],
)
def test_rnn_module(input_size, hidden_size, sequence_len, epochs):
    """Test that PyNeuraLogic RNN layer computes the same as PyTorch RNN layer (with backprop)"""
    torch_input = torch.randn((sequence_len, input_size))
    h0 = torch.randn((1, hidden_size))
    target = torch.randn((hidden_size,))

    rnn = torch.nn.RNN(input_size, hidden_size, 1, bias=False)

    template = Template()
    template += RNN(input_size, hidden_size, "h", "f", "h0", arity=0)

    model = template.build(
        Settings(chain_pruning=False, iso_value_compression=False, optimizer=Adam(lr=0.001), error_function=MSE())
    )

    parameters = model.parameters()
    torch_parameters = [parameter.tolist() for parameter in rnn.parameters()]

    parameters["weights"][0] = torch_parameters[0]
    parameters["weights"][1] = torch_parameters[1]

    model.load_state_dict(parameters)

    dataset = Dataset(
        [
            Sample(
                R.h(sequence_len)[target.detach().numpy().tolist()],
                [
                    R.h0[[float(h) for h in h0[0]]],
                    *[R.f(i + 1)[[float(h) for h in torch_input[i]]] for i in range(sequence_len)],
                ],
            ),
        ]
    )

    bd = model.build_dataset(dataset)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    loss_fun = torch.nn.MSELoss()

    for _ in range(epochs):
        output, _ = rnn(torch_input, h0)
        loss = loss_fun(output[-1], target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        result, _ = model(bd.samples)
        assert np.allclose([float(x) for x in output[-1]], [float(x) for x in result[0][1]], atol=10e-5)


@pytest.mark.parametrize(
    "input_size, hidden_size, sequence_len, epochs",
    [
        (10, 5, 10, 500),
    ],
)
def test_lstm_module(input_size, hidden_size, sequence_len, epochs):
    """Test that PyNeuraLogic LSTM layer computes the same as PyTorch LSTM layer (with backprop)"""
    torch_input = torch.randn((sequence_len, input_size))
    h0 = torch.randn((1, hidden_size))
    c0 = torch.randn((1, hidden_size))
    target = torch.randn((hidden_size,))

    rnn = torch.nn.LSTM(input_size, hidden_size, 1, bias=False)

    template = Template()
    template += LSTM(input_size, hidden_size, "h", "f", "h0", "c0", arity=0)

    model = template.build(
        Settings(chain_pruning=False, iso_value_compression=False, optimizer=Adam(lr=0.001), error_function=MSE())
    )

    parameters = model.parameters()
    torch_parameters = [parameter.tolist() for parameter in rnn.parameters()]

    parameters["weights"][0] = [torch_parameters[0][i] for i in range(0, hidden_size)]
    parameters["weights"][2] = [torch_parameters[0][i] for i in range(1 * hidden_size, 1 * hidden_size + hidden_size)]
    parameters["weights"][4] = [torch_parameters[0][i] for i in range(3 * hidden_size, 3 * hidden_size + hidden_size)]
    parameters["weights"][6] = [torch_parameters[0][i] for i in range(2 * hidden_size, 2 * hidden_size + hidden_size)]

    parameters["weights"][1] = [torch_parameters[1][i] for i in range(0, hidden_size)]
    parameters["weights"][3] = [torch_parameters[1][i] for i in range(1 * hidden_size, 1 * hidden_size + hidden_size)]
    parameters["weights"][5] = [torch_parameters[1][i] for i in range(3 * hidden_size, 3 * hidden_size + hidden_size)]
    parameters["weights"][7] = [torch_parameters[1][i] for i in range(2 * hidden_size, 2 * hidden_size + hidden_size)]

    model.load_state_dict(parameters)

    dataset = Dataset(
        [
            Sample(
                R.h(sequence_len)[target.detach().numpy().tolist()],
                [
                    R.c0[[float(c) for c in c0[0]]],
                    R.h0[[float(h) for h in h0[0]]],
                    *[R.f(i + 1)[[float(h) for h in torch_input[i]]] for i in range(sequence_len)],
                ],
            ),
        ]
    )

    bd = model.build_dataset(dataset)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    loss_fun = torch.nn.MSELoss()

    for _ in range(epochs):
        output, _ = rnn(torch_input, (h0, c0))
        loss = loss_fun(output[-1], target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        result, _ = model(bd.samples)
        assert np.allclose([float(x) for x in output[-1]], [float(x) for x in result[0][1]], atol=10e-5)
