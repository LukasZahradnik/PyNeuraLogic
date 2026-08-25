"""Tests for the static graph dataset mode."""

from neuralogic.dataset import Dataset, Sample
from neuralogic.core import Model, Settings, Relation, Var
from neuralogic.core.builder.static_graph import StaticGraphDataset, build_static_graph_dataset
from neuralogic.nn.optim import SGD


def test_static_graph_build():
    """Test that a StaticGraphDataset can be built from a Dataset."""
    dataset = Dataset()

    example_1 = [
        Relation.edge(0, 1),
        Relation.edge(1, 0),
        Relation.node_feature(0)[1.0],
        Relation.node_feature(1)[-1.0],
    ]
    example_2 = [
        Relation.edge(0, 1),
        Relation.edge(1, 0),
        Relation.node_feature(0)[2.0],
        Relation.node_feature(1)[0.5],
    ]

    dataset.add_samples([
        Sample(Relation.predict(0)[1.0], example_1),
        Sample(Relation.predict(0)[0.0], example_2),
    ])

    # Build a simple model
    model = Model()
    model.add_rule(Relation.predict(Var.X)[1.0] <= (Relation.edge(Var.X, Var.Y), Relation.node_feature(Var.Y)))

    settings = Settings(optimizer=SGD(0.1))
    model.build(settings)

    static_dataset = model.build_static_dataset(dataset)

    assert isinstance(static_dataset, StaticGraphDataset)
    assert len(static_dataset) == 2
    assert static_dataset.static_sample is not None
    assert len(static_dataset.fact_mappings) == 2

    # Check fact mappings
    for mapping in static_dataset.fact_mappings:
        # Each example has 4 facts
        assert len(mapping) == 4


def test_static_graph_train():
    """Test training with StaticGraphDataset produces results."""
    dataset = Dataset()

    # Shared structure: two nodes connected by edges
    shared = [
        Relation.edge(0, 1),
        Relation.edge(1, 0),
    ]

    # Different feature values per sample
    dataset.add_samples([
        Sample(
            Relation.predict(0)[1.0],
            shared + [Relation.node_feature(0)[1.0], Relation.node_feature(1)[-1.0]],
        ),
        Sample(
            Relation.predict(0)[0.0],
            shared + [Relation.node_feature(0)[-1.0], Relation.node_feature(1)[1.0]],
        ),
        Sample(
            Relation.predict(0)[1.0],
            shared + [Relation.node_feature(0)[1.0], Relation.node_feature(1)[-1.0]],
        ),
        Sample(
            Relation.predict(0)[0.0],
            shared + [Relation.node_feature(0)[-1.0], Relation.node_feature(1)[1.0]],
        ),
    ])

    # Build model: predict(X) = sum over neighbors of node_feature(Y)
    model = Model()
    model.add_rule(Relation.predict(Var.X)[1.0] <= (Relation.edge(Var.X, Var.Y), Relation.node_feature(Var.Y)))

    settings = Settings(optimizer=SGD(0.1))
    model.build(settings)

    static_dataset = model.build_static_dataset(dataset)

    # Train with static graph
    output = model.train(static_dataset, epochs=10)
    assert len(output) == 4 * 10  # 4 samples * 10 epochs


def test_static_graph_test():
    """Test evaluation with StaticGraphDataset."""
    dataset = Dataset()

    shared = [
        Relation.edge(0, 1),
        Relation.edge(1, 0),
    ]

    dataset.add_samples([
        Sample(
            Relation.predict(0)[1.0],
            shared + [Relation.node_feature(0)[1.0], Relation.node_feature(1)[-1.0]],
        ),
        Sample(
            Relation.predict(0)[0.0],
            shared + [Relation.node_feature(0)[-1.0], Relation.node_feature(1)[1.0]],
        ),
    ])

    model = Model()
    model.add_rule(Relation.predict(Var.X)[1.0] <= (Relation.edge(Var.X, Var.Y), Relation.node_feature(Var.Y)))

    settings = Settings(optimizer=SGD(0.1))
    model.build(settings)

    static_dataset = model.build_static_dataset(dataset)

    # Test (evaluation only, no weight update)
    results = model.test(static_dataset)
    assert len(results) == 2


def test_static_graph_vs_normal_equivalence():
    """Test that static graph training produces the same architecture as normal training.

    Both should produce the same number of results per epoch.
    """
    dataset = Dataset()

    shared = [
        Relation.edge(0, 1),
        Relation.edge(1, 0),
    ]

    dataset.add_samples([
        Sample(
            Relation.predict(0)[1.0],
            shared + [Relation.node_feature(0)[1.0], Relation.node_feature(1)[-1.0]],
        ),
        Sample(
            Relation.predict(0)[0.0],
            shared + [Relation.node_feature(0)[-1.0], Relation.node_feature(1)[1.0]],
        ),
    ])

    # Model 1: normal training
    model1 = Model()
    model1.add_rule(Relation.predict(Var.X)[1.0] <= (Relation.edge(Var.X, Var.Y), Relation.node_feature(Var.Y)))
    settings1 = Settings(optimizer=SGD(0.1))
    model1.build(settings1)
    built1 = model1.build_dataset(dataset)
    output1 = model1.train(built1, epochs=5)
    assert len(output1) == 2  # learnSamples returns results per sample for 1 epoch

    # Model 2: static graph training
    model2 = Model()
    model2.add_rule(Relation.predict(Var.X)[1.0] <= (Relation.edge(Var.X, Var.Y), Relation.node_feature(Var.Y)))
    settings2 = Settings(optimizer=SGD(0.1))
    model2.build(settings2)
    static_dataset2 = model2.build_static_dataset(dataset)
    output2 = model2.train(static_dataset2, epochs=5)
    assert len(output2) == 2 * 5  # 2 samples * 5 epochs


def test_static_graph_with_trainer():
    """Test using StaticGraphDataset with the Trainer API."""
    from neuralogic.nn.trainer import Trainer

    dataset = Dataset()

    shared = [
        Relation.edge(0, 1),
        Relation.edge(1, 0),
    ]

    dataset.add_samples([
        Sample(
            Relation.predict(0)[1.0],
            shared + [Relation.node_feature(0)[1.0], Relation.node_feature(1)[-1.0]],
        ),
        Sample(
            Relation.predict(0)[0.0],
            shared + [Relation.node_feature(0)[-1.0], Relation.node_feature(1)[1.0]],
        ),
    ])

    model = Model()
    model.add_rule(Relation.predict(Var.X)[1.0] <= (Relation.edge(Var.X, Var.Y), Relation.node_feature(Var.Y)))

    settings = Settings(optimizer=SGD(0.1))
    model.build(settings)

    static_dataset = model.build_static_dataset(dataset)

    trainer = Trainer(model)
    history = trainer.fit(static_dataset, epochs=5, silent=True)

    assert len(history.train_losses) == 5
