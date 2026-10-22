import pytest

pytest.importorskip("compute_graph_vectorize")
pytest.importorskip("torch")

import torch

from neuralogic.core import Model, R, Settings
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.module import Linear


def _build_linear_model_and_dataset():
    model = Model()
    model += Linear(2, 1, "h", "f")
    model = model.build(Settings(iso_value_compression=False, chain_pruning=False))

    dataset = Dataset(
        [
            Sample(R.h(0), [R.f(0)[[0.5, -0.2]]]),
            Sample(R.h(1), [R.f(1)[[1.0, 0.3]]]),
        ]
    )

    return model, dataset


def test_vectorize_returns_torch_module():
    model, dataset = _build_linear_model_and_dataset()

    vectorized = model.vectorize(dataset)

    assert isinstance(vectorized, torch.nn.Module)


def test_vectorized_output_matches_java_test():
    model, dataset = _build_linear_model_and_dataset()

    vectorized = model.vectorize(dataset)
    vectorized_output = vectorized(0).detach().flatten()

    java_output = torch.tensor(model.test(dataset), dtype=torch.float32)

    assert torch.allclose(vectorized_output, java_output, atol=1e-5)
    assert torch.isfinite(vectorized_output).all()


def test_vectorize_requires_built_model():
    model = Model()

    with pytest.raises(ValueError, match="model is not built"):
        model.vectorize(Dataset())
