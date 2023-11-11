import sys

from neuralogic.nn import get_evaluator
from neuralogic.utils.visualize import draw_model, draw_sample
from neuralogic.utils.data import XOR_Vectorized
from neuralogic.core import Settings


def test_draw_model():
    template, dataset = XOR_Vectorized()
    model = template.build(Settings())

    result = draw_model(model, show=False)

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_draw_sample():
    template, dataset = XOR_Vectorized()
    model = template.build(Settings())

    built_dataset = model.build_dataset(dataset)
    result = draw_sample(built_dataset.samples[0], show=False)

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_draw_model_from_template():
    template, dataset = XOR_Vectorized()
    result = template.draw(show=False)

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_draw_model_from_evaluator():
    template, dataset = XOR_Vectorized()

    evaluator = get_evaluator(template)
    result = evaluator.draw(show=False)

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_draw_sample_from_raw_sample():
    template, dataset = XOR_Vectorized()
    model = template.build(Settings())

    built_dataset = model.build_dataset(dataset)
    result = built_dataset.samples[0].draw(show=False)

    assert isinstance(result, bytes)
    assert len(result) > 0
