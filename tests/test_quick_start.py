from neuralogic.dataset import Data, TensorDataset, Dataset, Sample
from neuralogic.core import Template, Settings, Relation
from neuralogic.nn import get_evaluator
from neuralogic.nn.module import GCNConv
from neuralogic.optim import SGD


def test_quick_start_from_tensor():
    data = Data(
        edge_index=[
            [0, 1, 1, 2, 2, 0],
            [1, 0, 2, 1, 0, 2],
        ],
        x=[[0], [1], [-1]],
        y=[[1], [0], [1]],
        y_mask=[0, 1, 2],
    )

    dataset = TensorDataset(data=[data])

    assert len(dataset.data) == 1

    logic_dataset = dataset.to_dataset()

    assert len(logic_dataset) == 3
    assert len(logic_dataset[0]) == 9

    expected = [
        "<1> edge(0, 1).",
        "<1> edge(1, 0).",
        "<1> edge(1, 2).",
        "<1> edge(2, 1).",
        "<1> edge(2, 0).",
        "<1> edge(0, 2).",
        "<0> node_feature(0).",
        "<1> node_feature(1).",
        "<-1> node_feature(2).",
    ]

    for a, b in zip(logic_dataset[0].example, expected):
        assert str(a) == b

    assert str(logic_dataset[0]) == "1.0 predict(0)."
    assert str(logic_dataset[1]) == "0.0 predict(1)."
    assert str(logic_dataset[2]) == "1.0 predict(2)."


def test_model_evaluation_from_tensor():
    data = Data(
        edge_index=[
            [0, 1, 1, 2, 2, 0],
            [1, 0, 2, 1, 0, 2],
        ],
        x=[[0], [1], [-1]],
        y=[[1], [0], [1]],
        y_mask=[0, 1, 2],
    )

    dataset = TensorDataset(data=[data])

    template = Template()
    template.add_module(
        GCNConv(in_channels=1, out_channels=5, output_name="h0", feature_name="node_feature", edge_name="edge")
    )
    template.add_module(
        GCNConv(in_channels=5, out_channels=1, output_name="predict", feature_name="h0", edge_name="edge")
    )

    settings = Settings(optimizer=SGD(0.01), epochs=100)
    model = template.build(settings)
    built_dataset = model.build_dataset(dataset)

    model.train()  # or model.test() to change the mode
    output = model(built_dataset)

    assert len(output[0]) == 3
    assert output[1] == 3


def test_model_evaluation_from_logic():
    dataset = Dataset()

    example = [
        Relation.edge(0, 1),
        Relation.edge(1, 2),
        Relation.edge(2, 0),
        Relation.edge(1, 0),
        Relation.edge(2, 1),
        Relation.edge(0, 2),
        Relation.node_feature(0)[0],
        Relation.node_feature(1)[1],
        Relation.node_feature(2)[-1],
    ]

    dataset.add_samples(
        [
            Sample(Relation.predict(0)[1], example),
            Sample(Relation.predict(1)[0], example),
            Sample(Relation.predict(2)[1], example),
        ]
    )

    template = Template()
    template.add_module(
        GCNConv(in_channels=1, out_channels=5, output_name="h0", feature_name="node_feature", edge_name="edge")
    )
    template.add_module(
        GCNConv(in_channels=5, out_channels=1, output_name="predict", feature_name="h0", edge_name="edge")
    )

    settings = Settings(optimizer=SGD(0.01), epochs=100)
    model = template.build(settings)
    built_dataset = model.build_dataset(dataset)

    model.train()  # or model.test() to change the mode
    output = model(built_dataset)

    assert len(output[0]) == 3
    assert output[1] == 3


def test_evaluator_from_logic():
    dataset = Dataset()

    example = [
        Relation.edge(0, 1),
        Relation.edge(1, 2),
        Relation.edge(2, 0),
        Relation.edge(1, 0),
        Relation.edge(2, 1),
        Relation.edge(0, 2),
        Relation.node_feature(0)[0],
        Relation.node_feature(1)[1],
        Relation.node_feature(2)[-1],
    ]

    dataset.add_samples(
        [
            Sample(Relation.predict(0)[1], example),
            Sample(Relation.predict(1)[0], example),
            Sample(Relation.predict(2)[1], example),
        ]
    )

    template = Template()
    template.add_module(
        GCNConv(in_channels=1, out_channels=5, output_name="h0", feature_name="node_feature", edge_name="edge")
    )
    template.add_module(
        GCNConv(in_channels=5, out_channels=1, output_name="predict", feature_name="h0", edge_name="edge")
    )

    settings = Settings(optimizer=SGD(0.01), epochs=100)
    evaluator = get_evaluator(template, settings=settings)

    built_dataset = evaluator.build_dataset(dataset)
    output = evaluator.train(built_dataset, generator=False)

    assert len(output) == 2
    assert output[1] == 3
