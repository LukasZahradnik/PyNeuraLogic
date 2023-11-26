import pytest

import torch

from neuralogic.core import Template, R, Settings, Transformation
from neuralogic.nn.module import Linear
from neuralogic.dataset import Dataset, Sample


@pytest.mark.parametrize(
    "feature_size, output_size, num_of_inputs, activation",
    (
        (2, 3, 5, Transformation.IDENTITY),
        (3, 2, 2, Transformation.IDENTITY),
        (3, 3, 1, Transformation.IDENTITY),
        (3, 3, 1, Transformation.TANH),
    ),
)
def test_linear_module(feature_size: int, output_size: int, num_of_inputs: int, activation: Transformation):
    """Test that PyNeuraLogic linear layer computes the same as pytorch linear layer"""
    torch.manual_seed(1)
    linear_input = torch.rand((num_of_inputs, feature_size))
    linear = torch.nn.Linear(feature_size, output_size, bias=False)
    linear_output = linear(linear_input)

    if activation == Transformation.TANH:
        linear_output = torch.tanh_(linear_output)

    template = Template()
    template += Linear(feature_size, output_size, "h", "f", activation=activation)
    model = template.build(Settings(iso_value_compression=False, chain_pruning=False))

    state = model.state_dict()
    state["weights"][0] = list(linear.parameters())[0].detach().numpy()
    model.load_state_dict(state)

    samples = [Sample(R.h(index), [R.f(index)[row.detach().numpy()]]) for index, row in enumerate(linear_input)]
    built_dataset = model.build_dataset(Dataset(samples))

    for sample, row in zip(built_dataset.samples, linear_output):
        results = model(sample, train=False)

        assert len(results) == len(row)

        for a, b in zip(results, row):
            assert round(a, 3) == round(float(b), 3)
