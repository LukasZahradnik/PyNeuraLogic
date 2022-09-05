import pytest

import torch
import numpy as np

import neuralogic.nn.functional as F
from neuralogic.core import Template, R, Settings
from neuralogic.dataset import Dataset


@pytest.mark.parametrize(
    "torch_fun, fun",
    [
        (torch.sigmoid, F.sigmoid),
        (torch.tanh, F.tanh),
        (torch.sign, F.signum),
        (torch.relu, F.relu),
        (torch.exp, F.exp),
        (torch.log, F.log),
    ],
)
def test_transformation_body_function(torch_fun, fun):
    data = torch.rand((2, 3))

    torch_result = torch_fun(data).detach().numpy().round(3)

    template = Template()
    template += (R.h <= fun(R.input)) | [F.identity]
    template += R.h / 0 | [F.identity]

    model = template.build(Settings(iso_value_compression=False, chain_pruning=False))
    dataset = Dataset([[R.input[data.tolist()]]], [R.h])

    built_dataset = model.build_dataset(dataset)

    results = np.array(model(built_dataset, train=False)[0]).round(3)

    assert np.allclose(torch_result, results, atol=0.0001)
