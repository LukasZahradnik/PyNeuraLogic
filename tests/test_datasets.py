import numpy as np
import torch

from neuralogic.dataset import Data


def test_data_from_python_lists() -> None:
    """Tests data creation from python lists and scalars"""
    features = [[0, 1], [1, 0], [1, 0], [0, 1]]
    edges = [[0, 2], [1, 3]]

    y = 1

    data = Data(x=features, edge_index=edges, y=y)
    query, example = data.to_logic_form()

    expected_examples = [
        "<1> edge(0, 1).",
        "<1> edge(2, 3).",
        "<[0, 1]> node_feature(0).",
        "<[1, 0]> node_feature(1).",
        "<[1, 0]> node_feature(2).",
        "<[0, 1]> node_feature(3).",
    ]

    assert str(query) == "1.0 predict."
    for a, b in zip(expected_examples, example):
        assert str(a) == str(b)

    y = [2]

    data = Data(x=features, edge_index=edges, y=y)
    query, _ = data.to_logic_form()
    assert str(query) == "2.0 predict."

    query, _ = data.to_logic_form(one_hot_encode_labels=True, max_classes=3)
    assert str(query) == "[0.0, 0.0, 1.0] predict."

    y = [0, 1]

    data = Data(x=features, edge_index=edges, y=y)
    query, example = data.to_logic_form(
        feature_name="feat", edge_name="bond", output_name="out", one_hot_decode_features=True
    )

    expected_examples = [
        "<1> bond(0, 1).",
        "<1> bond(2, 3).",
        "<1> feat_1(0).",
        "<1> feat_0(1).",
        "<1> feat_0(2).",
        "<1> feat_1(3).",
    ]

    assert str(query) == "[0, 1] out."
    for a, b in zip(expected_examples, example):
        assert str(a) == str(b)

    data = Data(x=features, edge_index=edges, y=[0, 1, 2, 3], y_mask=[1, 2])
    query, _ = data.to_logic_form()

    assert len(query) == 2
    assert str(query[0]) == "1.0 predict(1)."
    assert str(query[1]) == "2.0 predict(2)."


def test_data_from_numpy() -> None:
    """Tests data creation from numpy"""
    features = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])
    edges = np.array([[0, 2], [1, 3]])

    y = np.array(1)

    data = Data(x=features, edge_index=edges, y=y)
    query, example = data.to_logic_form()

    expected_examples = [
        "<1> edge(0, 1).",
        "<1> edge(2, 3).",
        "<[0, 1]> node_feature(0).",
        "<[1, 0]> node_feature(1).",
        "<[1, 0]> node_feature(2).",
        "<[0, 1]> node_feature(3).",
    ]

    assert str(query) == "1.0 predict."
    for a, b in zip(expected_examples, example):
        assert str(a) == str(b)

    y = np.array([2])

    data = Data(x=features, edge_index=edges, y=y)
    query, _ = data.to_logic_form()
    assert str(query) == "2.0 predict."

    query, _ = data.to_logic_form(one_hot_encode_labels=True, max_classes=3)
    assert str(query) == "[0.0, 0.0, 1.0] predict."

    y = np.array([0, 1])

    data = Data(x=features, edge_index=edges, y=y)
    query, example = data.to_logic_form(
        feature_name="feat", edge_name="bond", output_name="out", one_hot_decode_features=True
    )

    expected_examples = [
        "<1> bond(0, 1).",
        "<1> bond(2, 3).",
        "<1> feat_1(0).",
        "<1> feat_0(1).",
        "<1> feat_0(2).",
        "<1> feat_1(3).",
    ]

    assert str(query) == "[0, 1] out."
    for a, b in zip(expected_examples, example):
        assert str(a) == str(b)

    data = Data(x=features, edge_index=edges, y=np.array([0, 1, 2, 3]), y_mask=np.array([1, 2]))
    query, _ = data.to_logic_form()

    assert len(query) == 2
    assert str(query[0]) == "1.0 predict(1)."
    assert str(query[1]) == "2.0 predict(2)."


def test_data_from_torch() -> None:
    """Tests data creation from torch"""
    features = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]])
    edges = torch.tensor([[0, 2], [1, 3]])

    y = torch.tensor(1)

    data = Data(x=features, edge_index=edges, y=y)
    query, example = data.to_logic_form()

    expected_examples = [
        "<1> edge(0, 1).",
        "<1> edge(2, 3).",
        "<[0, 1]> node_feature(0).",
        "<[1, 0]> node_feature(1).",
        "<[1, 0]> node_feature(2).",
        "<[0, 1]> node_feature(3).",
    ]

    assert str(query) == "1.0 predict."
    for a, b in zip(expected_examples, example):
        assert str(a) == str(b)

    y = torch.tensor([2])

    data = Data(x=features, edge_index=edges, y=y)
    query, _ = data.to_logic_form()
    assert str(query) == "2.0 predict."

    query, _ = data.to_logic_form(one_hot_encode_labels=True, max_classes=3)
    assert str(query) == "[0.0, 0.0, 1.0] predict."

    y = torch.tensor([0, 1])

    data = Data(x=features, edge_index=edges, y=y)
    query, example = data.to_logic_form(
        feature_name="feat", edge_name="bond", output_name="out", one_hot_decode_features=True
    )

    expected_examples = [
        "<1> bond(0, 1).",
        "<1> bond(2, 3).",
        "<1> feat_1(0).",
        "<1> feat_0(1).",
        "<1> feat_0(2).",
        "<1> feat_1(3).",
    ]

    assert str(query) == "[0, 1] out."
    for a, b in zip(expected_examples, example):
        assert str(a) == str(b)

    data = Data(x=features, edge_index=edges, y=torch.tensor([0, 1, 2, 3]), y_mask=torch.tensor([1, 2]))
    query, _ = data.to_logic_form()

    assert len(query) == 2
    assert str(query[0]) == "1.0 predict(1)."
    assert str(query[1]) == "2.0 predict(2)."


def test_data_edge_features_from_torch() -> None:
    """Tests data creation with edge features from torch"""

    features = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]])
    edges = torch.tensor([[0, 2], [1, 3]])
    edge_features = torch.tensor([[0, 1, 2], [3, 4, 5]])

    data = Data(x=features, edge_index=edges, edge_attr=edge_features, y=0)
    query, example = data.to_logic_form()

    expected_examples = [
        "<[0, 1, 2]> edge(0, 1).",
        "<[3, 4, 5]> edge(2, 3).",
        "<[0, 1]> node_feature(0).",
        "<[1, 0]> node_feature(1).",
        "<[1, 0]> node_feature(2).",
        "<[0, 1]> node_feature(3).",
    ]

    for a, b in zip(expected_examples, example):
        assert str(a) == str(b)


def test_data_edge_features_types() -> None:
    """Tests data creation with edge features from torch"""

    features = torch.tensor([])
    edges = torch.tensor([[0, 2], [1, 3]])
    edge_features = torch.tensor([[0, 1, 0], [1, 0, 0]])

    data = Data(x=features, edge_index=edges, edge_attr=edge_features, y=0)
    query, example = data.to_logic_form(one_hot_decode_edge_features=True)

    expected_examples = [
        "<1> edge_1(0, 1).",
        "<1> edge_0(2, 3).",
    ]

    for a, b in zip(expected_examples, example):
        assert str(a) == str(b)


def test_data_edge_features_from_numpy() -> None:
    """Tests data creation with edge features from numpy"""

    features = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])
    edges = np.array([[0, 2], [1, 3]])
    edge_features = np.array([[0, 1, 2], [3, 4, 5]])

    data = Data(x=features, edge_index=edges, edge_attr=edge_features, y=0)
    query, example = data.to_logic_form()

    expected_examples = [
        "<[0, 1, 2]> edge(0, 1).",
        "<[3, 4, 5]> edge(2, 3).",
        "<[0, 1]> node_feature(0).",
        "<[1, 0]> node_feature(1).",
        "<[1, 0]> node_feature(2).",
        "<[0, 1]> node_feature(3).",
    ]

    for a, b in zip(expected_examples, example):
        assert str(a) == str(b)


def test_data_edge_features_from_python_lists() -> None:
    """Tests data creation with edge features from python lists"""

    features = [[0, 1], [1, 0], [1, 0], [0, 1]]
    edges = [[0, 2], [1, 3]]
    edge_features = [[0, 1, 2], [3, 4, 5]]

    data = Data(x=features, edge_index=edges, edge_attr=edge_features, y=0)
    query, example = data.to_logic_form()

    expected_examples = [
        "<[0, 1, 2]> edge(0, 1).",
        "<[3, 4, 5]> edge(2, 3).",
        "<[0, 1]> node_feature(0).",
        "<[1, 0]> node_feature(1).",
        "<[1, 0]> node_feature(2).",
        "<[0, 1]> node_feature(3).",
    ]

    for a, b in zip(expected_examples, example):
        assert str(a) == str(b)
