from typing import Dict, Any

from neuralogic.core import Settings, Transformation
from neuralogic.core.constructs.function import Function
from neuralogic.nn.init import Initializer, Uniform
from neuralogic.nn.loss import SoftEntropy, ErrorFunction
from neuralogic.optim import SGD, Optimizer

import pytest


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "optimizer": SGD(0.5),
            "epochs": 100,
            "error_function": SoftEntropy(),
            "initializer": Uniform(5.0),
            "initializer_uniform_scale": 5.0,
            "rule_transformation": Transformation.SIGMOID,
            "relation_transformation": Transformation.RELU,
        }
    ],
)
def test_settings_proxy_properties_setting(parameters: Dict[str, Any]) -> None:
    """Tests propagation of changes on settings to its proxies"""
    settings = Settings()
    settings_proxy = settings.create_proxy()

    for key, value in parameters.items():
        if key == "initializer_uniform_scale":
            assert settings.initializer.scale == settings_proxy.__getattribute__(key)
            continue

        if isinstance(value, (int, float)):
            assert settings.__getattribute__(key) == settings_proxy.__getattribute__(key)
        elif isinstance(settings.__getattribute__(key), (ErrorFunction, Initializer, Function, Optimizer)):
            assert str(settings.__getattribute__(key)) == str(settings_proxy.__getattribute__(key))
        else:
            assert settings.__getattribute__(key) == str(settings_proxy.__getattribute__(key))

    for key, value in parameters.items():
        settings.__setattr__(key, value)

    for key, value in parameters.items():
        if key == "initializer_uniform_scale":
            assert settings.initializer.scale == settings_proxy.__getattribute__(key)
            continue

        if isinstance(value, (int, float)):
            assert settings.__getattribute__(key) == settings_proxy.__getattribute__(key)
        elif isinstance(settings.__getattribute__(key), (ErrorFunction, Initializer, Function, Optimizer)):
            assert str(settings.__getattribute__(key)) == str(settings_proxy.__getattribute__(key))
        else:
            assert settings.__getattribute__(key) == str(settings_proxy.__getattribute__(key))


def test_settings_proxies_creation() -> None:
    """Tests settings proxies creation and dereferencing"""
    settings = Settings()

    settings_proxy = settings.create_proxy()  # We have one proxy
    assert len(settings._proxies) == 1

    second_proxy = settings.create_proxy()  # Now we have two proxies
    assert len(settings._proxies) == 2

    # We are creating the third one, but the original `settings_proxy` is dereferenced
    settings_proxy = settings.create_proxy()
    assert len(settings._proxies) == 2

    del second_proxy
    assert len(settings._proxies) == 1

    del settings_proxy
    assert len(settings._proxies) == 0
