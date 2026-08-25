"""A learnable value on an example fact is a parameter of the *data*, and has to be savable as one.

It trains, but its weight is created per example after the model, so it is not in the model's `state_dict`
and saving the model does not save it. `BuiltDataset.state_dict()` is where it lives instead.
"""
import pytest

from neuralogic.core import Combination, Model, R, Settings, Transformation, V
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.loss import MSE
from neuralogic.nn.optim import SGD

WEIGHT = [[0.5, -0.5]]
FACTS = {"a": [0.3, 0.7], "b": [0.1, 0.9]}
TARGETS = {"a": 1.0, "b": 0.4}


def _model():
    model = Model()
    model += (R.out(V.X)["w":1, 2] <= R.emb(V.X)) | [Combination.SUM, Transformation.IDENTITY]
    model += R.out / 1 | [Transformation.IDENTITY]

    built = model.build(
        Settings(
            optimizer=SGD(lr=0.5),
            error_function=MSE(reduction="sum"),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    index = next(i for i, name in state["weight_names"].items() if str(name).strip() == "w")
    state["weights"][index] = WEIGHT
    built.load_state_dict(state)
    return built


def _dataset(built, learnable_facts=True):
    samples = [Sample(R.out(k)[TARGETS[k]], [R.emb(k)[FACTS[k]]]) for k in FACTS]
    return built.build_dataset(Dataset(samples), learnable_facts=learnable_facts)


def test_example_parameters_are_keyed_by_their_ground_literal():
    """The literal is the key, and it is the only candidate that survives a rebuild.

    The weight's index continues a counter that keeps running, so building the same dataset twice on one
    model hands the same parameters different indices - asserted here, because it is the reason the key is
    what it is - and the generated weight name is `w` plus that index, so it moves with it.
    """
    built = _model()

    first = _dataset(built)
    second = _dataset(built)

    assert first.state_dict()["weights"] == {"emb(a)": FACTS["a"], "emb(b)": FACTS["b"]}
    assert second.state_dict()["weights"] == first.state_dict()["weights"]

    indices = [int(w.index) for w in first._example_parameters().values()]
    rebuilt = [int(w.index) for w in second._example_parameters().values()]
    assert indices != rebuilt, "if indices were stable, keying by them would have been fine"


def test_a_trained_example_parameter_survives_a_save_and_restore():
    """The whole point: train, save both dicts, restore into a fresh model, get the same model back."""
    built = _model()
    data = _dataset(built)
    built.train(data, epochs=20)

    model_state, data_state = built.state_dict(), data.state_dict()
    trained = [float(output) for output in built(data)]

    fresh = _model()
    fresh.load_state_dict(model_state)
    fresh_data = _dataset(fresh)

    # the model alone is not enough - this is the state the defect left behind
    assert [float(o) for o in fresh(fresh_data)] != pytest.approx(trained, abs=1e-9)

    fresh_data.load_state_dict(data_state)
    assert [float(o) for o in fresh(fresh_data)] == pytest.approx(trained, abs=1e-12)


def test_the_values_actually_move_so_the_round_trip_is_not_trivial():
    """A parameter that never trained would round-trip by accident."""
    built = _model()
    data = _dataset(built)

    before = data.state_dict()["weights"]
    built.train(data, epochs=20)
    after = data.state_dict()["weights"]

    assert set(before) == set(after)
    for literal in before:
        assert after[literal] != pytest.approx(before[literal], abs=1e-9)


def test_nothing_to_save_without_learnable_facts():
    """The flag off means the example's values are constants, and constants are not parameters."""
    built = _model()
    assert _dataset(built, learnable_facts=False).state_dict()["weights"] == {}


def test_a_literal_the_dataset_does_not_have_is_refused():
    """Silently ignoring it would leave a model half restored and say nothing."""
    built = _model()
    data = _dataset(built)

    with pytest.raises(ValueError, match="no learnable example fact"):
        data.load_state_dict({"weights": {"emb(nowhere)": [1.0, 2.0]}})


def test_facts_sharing_one_named_weight_cannot_be_given_different_values():
    """A *named* weight written into several examples is one shared parameter, not one per example.

    It still appears under each literal, because that is where it is reachable from, but only one of two
    differing values could survive and which one is an accident of iteration order - so neither is written.
    """
    model = Model()
    model += (R.out(V.X)["w":1, 2] <= R.emb(V.X)) | [Combination.SUM, Transformation.IDENTITY]
    model += R.out / 1 | [Transformation.IDENTITY]
    built = model.build(
        Settings(
            # gentler than the other cases on purpose: here two multiplicative parameters train at once and
            # the pair diverges to NaN at the rate the rest of this file uses
            optimizer=SGD(lr=0.02),
            error_function=MSE(reduction="sum"),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    data = built.build_dataset(
        Dataset([Sample(R.out(k)[TARGETS[k]], [R.emb(k)["shared":2, ]]) for k in FACTS]),
        learnable_facts=True,
    )
    built.train(data, epochs=10)

    saved = data.state_dict()["weights"]
    assert saved["emb(a)"] == saved["emb(b)"], "one weight, so one value"

    data.load_state_dict(saved)  # the same value under both is what a save produces, and must work

    with pytest.raises(ValueError, match="share one weight"):
        data.load_state_dict({"weights": {"emb(a)": [1.0, 2.0], "emb(b)": [3.0, 4.0]}})


def test_a_learnable_template_fact_stays_the_models_and_is_not_reported_twice():
    """A fact declared in the *template* is a model parameter, and the model already saves it.

    Both kinds end up as fact neurons carrying a learnable weight, so the two are only told apart by which
    of the two builders made them. Without that, a template parameter would be reported by the model and by
    every dataset built from it, and restoring would write it from two places.
    """
    model = Model()
    model += R.emb("t")[[0.2, 0.8]]  # learnable by default, unlike the same thing inside an example
    model += (R.out(V.X)["w":1, 2] <= R.emb(V.X)) | [Combination.SUM, Transformation.IDENTITY]
    model += R.out / 1 | [Transformation.IDENTITY]
    built = model.build(
        Settings(
            optimizer=SGD(lr=0.05),
            error_function=MSE(reduction="sum"),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    data = built.build_dataset(
        Dataset(
            [
                Sample(R.out("t")[1.0], [R.exists("t")]),
                Sample(R.out("a")[0.4], [R.emb("a")[FACTS["a"]]]),
            ]
        ),
        learnable_facts=True,
    )

    names = {str(name).strip() for name in built.state_dict()["weight_names"].values()}
    assert "w" in names and len(names) == 2, "the template fact is one of the model's own parameters"
    assert data.state_dict()["weights"] == {"emb(a)": FACTS["a"]}, "and only the example's is the data's"
