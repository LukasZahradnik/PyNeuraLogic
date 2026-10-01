import networkx as nx

from neuralogic.utils.visualize import (
    draw_model,
    draw_sample,
    model_to_graphml_source,
    model_to_networkx,
    sample_to_graphml_source,
    sample_to_networkx,
    save_model_graphml,
    save_sample_graphml,
)
from neuralogic.utils.data import XOR_Vectorized
from neuralogic.core import Settings


def test_draw_model():
    model, dataset = XOR_Vectorized()
    model = model.build(Settings())

    result = draw_model(model, show=False)

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_draw_sample():
    model, dataset = XOR_Vectorized()
    model = model.build(Settings())

    built_dataset = model.build_dataset(dataset)
    result = draw_sample(built_dataset[0], show=False)

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_draw_sample_from_raw_sample():
    model, dataset = XOR_Vectorized()
    model = model.build(Settings())

    built_dataset = model.build_dataset(dataset)
    result = built_dataset[0].draw(show=False)

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_model_to_graphml_source():
    model, dataset = XOR_Vectorized()
    model = model.build(Settings())

    graphml = model_to_graphml_source(model)

    assert isinstance(graphml, str)
    assert graphml.strip().startswith("<?xml")


def test_sample_to_graphml_source():
    model, dataset = XOR_Vectorized()
    model = model.build(Settings())

    built_dataset = model.build_dataset(dataset)
    graphml = sample_to_graphml_source(built_dataset[0])

    assert isinstance(graphml, str)
    assert graphml.strip().startswith("<?xml")


def test_model_to_networkx():
    model, dataset = XOR_Vectorized()
    model = model.build(Settings())

    graph = model_to_networkx(model)

    assert isinstance(graph, nx.DiGraph)
    assert all("label" in data for _, data in graph.nodes(data=True))


def test_sample_to_networkx():
    model, dataset = XOR_Vectorized()
    model = model.build(Settings())

    built_dataset = model.build_dataset(dataset)
    graph = sample_to_networkx(built_dataset[0])

    assert isinstance(graph, nx.DiGraph)
    assert all("label" in data for _, data in graph.nodes(data=True))


def test_save_model_graphml(tmp_path):
    model, dataset = XOR_Vectorized()
    model = model.build(Settings())

    path = tmp_path / "model.graphml"
    save_model_graphml(model, str(path))

    assert path.exists()
    graph = nx.read_graphml(path)
    assert isinstance(graph, nx.DiGraph)


def test_save_sample_graphml(tmp_path):
    model, dataset = XOR_Vectorized()
    model = model.build(Settings())

    built_dataset = model.build_dataset(dataset)
    path = tmp_path / "sample.graphml"
    save_sample_graphml(built_dataset[0], str(path))

    assert path.exists()
    graph = nx.read_graphml(path)
    assert isinstance(graph, nx.DiGraph)
