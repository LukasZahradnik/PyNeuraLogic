import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GENConv, GINEConv, GCNConv

from neuralogic.core import Template, Settings, R, Aggregation
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.loss import MSE

import neuralogic.nn.module
from neuralogic.optim import Adam


@pytest.mark.parametrize(
    "input_size, hidden_size",
    [
        (5, 1),
        (10, 1),
    ],
)
def test_gen_module(input_size, hidden_size):
    """Test that PyNeuraLogic GENConv layer computes the same as PyG GENConv layer"""

    edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long)
    x = torch.randn(
        (
            3,
            input_size,
        ),
        dtype=torch.float,
    )
    e = torch.randn(
        (
            4,
            input_size,
        ),
        dtype=torch.float,
    )

    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=e)

    gen = GENConv(input_size, hidden_size, aggr="mean", num_layers=1, eps=0, edge_dim=input_size)
    for m in gen.mlp._modules.values():
        m.bias = None

    template = Template()
    template += neuralogic.nn.module.GENConv(
        input_size, hidden_size, "h", "f", "e", num_layers=1, aggregation=Aggregation.AVG, eps=0, edge_dim=input_size
    )

    model = template.build(
        Settings(chain_pruning=False, iso_value_compression=False, optimizer=Adam(lr=0.001), error_function=MSE())
    )

    parameters = model.parameters()
    torch_parameters = [parameter.tolist() for parameter in gen.parameters()]

    parameters["weights"][2] = [torch_parameters[0][i] for i in range(0, hidden_size)]
    parameters["weights"][1] = [torch_parameters[1][i] for i in range(0, hidden_size)]
    parameters["weights"][3] = [torch_parameters[2][i] for i in range(0, hidden_size)]
    parameters["weights"][4] = torch_parameters[3][0]

    model.load_state_dict(parameters)

    example = [
        R.f(0)[x[0]],
        R.f(1)[x[1]],
        R.f(2)[x[2]],
        R.e(0, 1)[e[0]],
        R.e(1, 0)[e[1]],
        R.e(1, 2)[e[2]],
        R.e(2, 1)[e[3]],
    ]

    dataset = Dataset([Sample(R.h(i), example) for i in range(3)])
    bd = model.build_dataset(dataset)

    output = gen(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
    result = model(bd.samples, train=False)

    assert np.allclose([float(x) for x in output], [float(x) for x in result], atol=10e-5)


@pytest.mark.parametrize(
    "input_size",
    [
        5,
        10,
    ],
)
def test_gine_module(input_size):
    """Test that PyNeuraLogic GINEConv layer computes the same as PyG GINEConv layer"""

    edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long)
    x = torch.randn(
        (
            3,
            input_size,
        ),
        dtype=torch.float,
    )
    e = torch.randn(
        (
            4,
            input_size,
        ),
        dtype=torch.float,
    )

    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=e)

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    gin = GINEConv(Identity())

    template = Template()
    template += neuralogic.nn.module.GINEConv(input_size, "f", "e", "h")

    model = template.build(
        Settings(chain_pruning=False, iso_value_compression=False, optimizer=Adam(lr=0.001), error_function=MSE())
    )

    example = [
        R.f(0)[x[0]],
        R.f(1)[x[1]],
        R.f(2)[x[2]],
        R.e(0, 1)[e[0]],
        R.e(1, 0)[e[1]],
        R.e(1, 2)[e[2]],
        R.e(2, 1)[e[3]],
    ]

    dataset = Dataset([Sample(R.h(i), example) for i in range(3)])
    bd = model.build_dataset(dataset)

    output = gin(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
    result = model(bd.samples, train=False)

    assert np.allclose([[float(x) for x in xs] for xs in output], [[float(x) for x in xs] for xs in result], atol=10e-5)


@pytest.mark.parametrize(
    "input_size, output_size",
    [
        (5, 2),
        (10, 2),
    ],
)
def test_gcn_module(input_size, output_size):
    """Test that PyNeuraLogic GCNConv layer computes the same as PyG GCNConv layer"""

    edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long)
    x = torch.randn(
        (
            3,
            input_size,
        ),
        dtype=torch.float,
    )
    e = torch.ones(
        (4,),
        dtype=torch.float,
    )

    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=e)
    gcn = GCNConv(input_size, output_size, bias=False)

    template = Template()
    template += neuralogic.nn.module.GCNConv(input_size, output_size, "h", "f", "e")

    model = template.build(
        Settings(chain_pruning=False, iso_value_compression=False, optimizer=Adam(lr=0.001), error_function=MSE())
    )

    parameters = model.parameters()
    torch_parameters = [parameter.tolist() for parameter in gcn.parameters()]

    parameters["weights"][1] = [torch_parameters[0][i] for i in range(0, output_size)]
    model.load_state_dict(parameters)

    example = [
        R.f(0)[x[0]],
        R.f(1)[x[1]],
        R.f(2)[x[2]],
        R.e(0, 1)[e[0]],
        R.e(1, 0)[e[1]],
        R.e(1, 2)[e[2]],
        R.e(2, 1)[e[3]],
    ]

    dataset = Dataset([Sample(R.h(i), example) for i in range(3)])
    bd = model.build_dataset(dataset)

    output = gcn(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr)
    result = model(bd.samples, train=False)

    assert np.allclose([[float(x) for x in xs] for xs in output], [[float(x) for x in xs] for xs in result], atol=10e-5)
