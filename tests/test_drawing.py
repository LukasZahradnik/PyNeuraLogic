import sys

from neuralogic.utils.visualize import draw_model, draw_sample
from neuralogic.utils.data import XOR_Vectorized
from neuralogic.core import Settings, Backend


def test_draw_model():
    template, dataset = XOR_Vectorized()
    model = template.build(Backend.JAVA, Settings())

    result = draw_model(model, draw_ipython=False)

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_draw_sample():
    template, dataset = XOR_Vectorized()
    model = template.build(Backend.JAVA, Settings())

    built_dataset = model.build_dataset(dataset)
    result = draw_sample(built_dataset.samples[0], draw_ipython=False)

    assert isinstance(result, bytes)
    assert len(result) > 0
