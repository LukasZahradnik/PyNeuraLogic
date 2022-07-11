import torch

from neuralogic.core import Template, Settings, R
from neuralogic.dataset import Dataset

from neuralogic.nn.module import GRU


def test_gru_module():
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
    template += GRU(in_size, hid_size, steps, "h", "f", "h0", arity=0)

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
