import pytest

import torch
import numpy as np

import neuralogic.nn.functional as F
from neuralogic.core import Template, R, Settings
from neuralogic.dataset import Dataset, Sample


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
    dataset = Dataset([Sample(R.h, R.input[data.tolist()])])

    built_dataset = model.build_dataset(dataset)

    results = np.array(model(built_dataset, train=False)[0]).round(3)

    assert np.allclose(torch_result, results, atol=0.0001)


def test_slice_function():
    data = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
    )

    res = np.array(
        [
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]
    )

    template = Template()
    template += (R.h <= F.slice(R.input, rows=(1, 3))) | [F.identity]
    template += R.h / 0 | [F.identity]

    model = template.build(Settings(iso_value_compression=False, chain_pruning=False))
    dataset = Dataset([Sample(R.h, [R.input[data]])])

    built_dataset = model.build_dataset(dataset)
    results = np.array(model(built_dataset, train=False)[0])

    assert np.allclose(res, results)

    template = Template()
    template += (R.h <= R.input) | [F.slice(rows=(1, 3))]
    template += R.h / 0 | [F.identity]

    model = template.build(Settings(iso_value_compression=False, chain_pruning=False))
    dataset = Dataset(Sample(R.h, [R.input[data]]))

    built_dataset = model.build_dataset(dataset)
    results = np.array(model(built_dataset, train=False)[0])

    assert np.allclose(res, results)

    template = Template()
    template += (R.h <= R.input) | [F.identity]
    template += R.h / 0 | [F.slice(rows=(1, 3))]

    model = template.build(Settings(iso_value_compression=False, chain_pruning=False))
    dataset = Dataset(Sample(R.h, [R.input[data]]))

    built_dataset = model.build_dataset(dataset)
    results = np.array(model(built_dataset, train=False)[0])

    assert np.allclose(res, results)
