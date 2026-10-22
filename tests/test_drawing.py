import networkx as nx

from neuralogic.utils.visualize import (
    DrawingSettings,
    NodeAttributes,
    NodeStyles,
    draw_model,
    draw_sample,
    get_grounding_drawer,
    model_to_dot_source,
    model_to_graphml_source,
    model_to_networkx,
    sample_to_graphml_source,
    sample_to_networkx,
    save_model_graphml,
    save_sample_graphml,
    to_dot_source,
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


def test_drawing_settings_constructor():
    settings = DrawingSettings(
        value_detail=1,
        value_decimals=4,
        display_values=False,
        display_weights=False,
        graph={"bgcolor": "white"},
        node={"fact": {"shape": "box", "fillcolor": "#fff2cc"}},
        edge={"ruleToHead": {"color": "red"}},
    )

    assert settings.value_decimals == 4
    assert settings.display_values is False
    assert settings.display_weights is False


def test_drawing_settings_constructor_attrs_are_applied():
    model, dataset = XOR_Vectorized()
    model = model.build(Settings())

    settings = DrawingSettings(graph={"bgcolor": "white"})
    dot = model_to_dot_source(model, settings)

    assert 'bgcolor="white"' in dot


def test_drawing_settings_constructor_with_style_objects():
    settings = DrawingSettings(node=NodeStyles(fact=NodeAttributes(color="#ff0000", shape="rect")))

    result = draw_model(XOR_Vectorized()[0].build(Settings()), show=False, settings=settings)

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_grounding_drawing_fact_style_alias():
    model, dataset = XOR_Vectorized()
    model = model.build(Settings())

    grounding = model.ground(dataset)[0]
    settings = DrawingSettings(node=NodeStyles(fact=NodeAttributes(color="#ff0000", shape="rect")))

    dot = to_dot_source(get_grounding_drawer(settings), grounding._grounding)

    assert "#ff0000" in dot
    assert 'shape="rect"' in dot


def test_draw_model_with_drawing_settings():
    model, dataset = XOR_Vectorized()
    model = model.build(Settings())

    settings = DrawingSettings(
        value_detail=1,
        display_values=False,
        node={"fact": {"shape": "box", "fillcolor": "#fff2cc"}},
    )

    result = draw_model(model, show=False, img_type="svg", settings=settings)

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_model_draw_method_with_drawing_settings():
    model, dataset = XOR_Vectorized()
    model = model.build(Settings())

    settings = DrawingSettings(value_decimals=4)

    result = model.draw(show=False, settings=settings)

    assert isinstance(result, bytes)
    assert len(result) > 0
