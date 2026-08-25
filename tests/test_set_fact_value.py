"""`set_fact_value` must not write through a weight the whole JVM shares.

A fact with no weight of its own is handed `Weight.zeroWeight` by `WeightedNeuron`'s constructor, and that is
a *static* field. Writing the fact's value through it replaced the logical zero for every model built
afterwards in the same process.
"""
import jpype
import pytest

from neuralogic.core import Combination, Model, R, Settings, Transformation, V
from neuralogic.core.builder.components import NeuronType
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.loss import MSE
from neuralogic.nn.optim import SGD


def _sample_with_an_unvalued_fact():
    model = Model()
    model += (R.out(V.X)["w":1, 1] <= (R.edge(V.X, V.Y), R.feat(V.Y))) | [
        Combination.SUM,
        Transformation.IDENTITY,
    ]
    model += R.out / 1 | [Transformation.IDENTITY]
    built = model.build(
        Settings(
            optimizer=SGD(lr=0.1),
            error_function=MSE(reduction="sum"),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    # `edge(0, 1)` carries no value, so its fact neuron gets the shared weight rather than one of its own
    data = built.build_dataset(Dataset([Sample(R.out(0)[1.0], [R.edge(0, 1), R.feat(1)[0.25]])]))
    return built, data._samples[0]


def _scalar(value):
    # set_fact_value reaches a Java setter that takes a Value, not a float, whatever its docstring says
    return jpype.JClass("cz.cvut.fel.ida.algebra.values.ScalarValue")(float(value))


def _shared():
    weight_class = jpype.JClass("cz.cvut.fel.ida.algebra.weights.Weight")
    return (
        float(weight_class.zeroWeight.value.get(0)),
        float(weight_class.unitWeight.value.get(0)),
    )


def test_setting_an_unvalued_facts_value_leaves_the_shared_weights_alone():
    """The case that broke a whole test suite once: the write landed on a static field."""
    built, sample = _sample_with_an_unvalued_fact()

    assert _shared() == (0.0, 1.0), "precondition: nothing has corrupted them yet"
    index = sample.set_fact_value(R.edge(0, 1), _scalar(7.0))

    assert index >= 0, "the fact was found, so the write really did happen"
    assert _shared() == (0.0, 1.0)


def test_the_value_still_reaches_the_fact():
    """Guarding the offset must not turn the whole call into a no-op."""
    built, sample = _sample_with_an_unvalued_fact()

    before = float(sample.get_neurons(R.edge(0, 1), NeuronType.Fact)[0].value)
    sample.set_fact_value(R.edge(0, 1), _scalar(7.0))
    after = float(sample.get_neurons(R.edge(0, 1), NeuronType.Fact)[0].value)

    assert after == pytest.approx(7.0)
    assert before != pytest.approx(7.0)
