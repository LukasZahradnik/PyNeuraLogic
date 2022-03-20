from typing import Dict, Any

import neuralogic.core.error_function
from neuralogic.core import Settings, Initializer, Optimizer, Activation
from neuralogic.core.error_function import SoftEntropy

import pytest


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "optimizer": Optimizer.SGD,
            "learning_rate": 0.5,
            "epochs": 100,
            "error_function": SoftEntropy(),
            "initializer": Initializer.NORMAL,
            "initializer_const": 1,
            "initializer_uniform_scale": 5.0,
            "rule_neuron_activation": Activation.SIGMOID,
            "relation_neuron_activation": Activation.RELU,
        }
    ],
)
def test_settings_proxy_properties_setting(parameters: Dict[str, Any]) -> None:
    """Tests propagation of changes on settings to its proxies"""
    settings = Settings()
    settings_proxy = settings.create_proxy()

    for key, value in parameters.items():
        if isinstance(value, (int, float)):
            assert settings.__getattribute__(key) == settings_proxy.__getattribute__(key)
        elif isinstance(settings.__getattribute__(key), neuralogic.core.error_function.ErrorFunction):
            assert str(settings.__getattribute__(key)) == str(settings_proxy.__getattribute__(key))
        else:
            assert settings.__getattribute__(key) == str(settings_proxy.__getattribute__(key))

    for key, value in parameters.items():
        settings.__setattr__(key, value)

    for key, value in parameters.items():
        if isinstance(value, (int, float)):
            assert settings.__getattribute__(key) == settings_proxy.__getattribute__(key)
        elif isinstance(settings.__getattribute__(key), neuralogic.core.error_function.ErrorFunction):
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
