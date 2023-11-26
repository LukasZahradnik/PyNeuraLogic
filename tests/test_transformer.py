import numpy as np
import pytest
import torch

from neuralogic.core import Template, Settings, V, R
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.module import MultiheadAttention


@pytest.mark.parametrize(
    "qdim, kdim, vdim, num_heads, sequence_len",
    [
        (10, 12, 14, 1, 3),  # One head
        (10, 12, 14, 2, 3),  # Two heads
    ],
)
def test_multiheadattention(qdim: int, kdim: int, vdim: int, num_heads: int, sequence_len: int):
    keys = torch.rand((sequence_len, kdim))
    queries = torch.rand((sequence_len, qdim))
    values = torch.rand((sequence_len, vdim))

    mha = torch.nn.MultiheadAttention(qdim, num_heads, bias=False, kdim=kdim, vdim=vdim)

    template = Template()
    template += MultiheadAttention(qdim, num_heads, "out", "q", "k", "v", vdim=vdim, kdim=kdim)

    model = template.build(Settings(iso_value_compression=False, chain_pruning=False))

    params = model.parameters()
    torch_params = list(mha.parameters())

    params["weights"][1] = torch_params[0].detach().numpy().tolist()
    params["weights"][2] = torch_params[2].detach().numpy().tolist()
    params["weights"][3] = torch_params[1].detach().numpy().tolist()
    params["weights"][4] = torch_params[3].detach().numpy().tolist()

    model.load_state_dict(params)

    example = []
    for i in range(sequence_len):
        example.append(R.v(i)[values[i]])
        example.append(R.k(i)[keys[i]])
        example.append(R.q(i)[queries[i]])

    dataset = Dataset([Sample(R.out(i), example) for i in range(sequence_len)])
    built_dataset = model.build_dataset(dataset)

    torch_results = mha(queries, keys, values)
    pyneuralogic_results = model(built_dataset, train=False)

    for t_res, pnl_res in zip(torch_results[0], pyneuralogic_results):
        assert np.allclose(t_res.detach().numpy(), np.array(pnl_res))
