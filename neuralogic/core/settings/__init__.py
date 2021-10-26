from typing import Any, Optional
import weakref

from neuralogic.core.settings.settings_proxy import SettingsProxy
from neuralogic.core.enums import Optimizer, Initializer, ErrorFunction, Activation


class Settings:
    def __init__(
        self,
        *,
        optimizer: Optimizer = Optimizer.ADAM,
        learning_rate: Optional[float] = None,
        epochs: int = 3000,
        error_function: ErrorFunction = ErrorFunction.SQUARED_DIFF,
        initializer: Initializer = Initializer.UNIFORM,
        initializer_const: float = 0.1,
        initializer_uniform_scale: float = 2.0,
        rule_neuron_activation: Activation = Activation.TANH,
        relation_neuron_activation: Activation = Activation.TANH,
    ):
        self.params = locals().copy()
        self.params.pop("self")
        self._proxies = weakref.WeakSet()

    @property
    def iso_value_compression(self) -> bool:
        return self.params["iso_value_compression"]

    @iso_value_compression.setter
    def iso_value_compression(self, iso_value_compression: bool):
        self._update("iso_value_compression", iso_value_compression)

    @property
    def seed(self) -> int:
        return self.params["seed"]

    @seed.setter
    def seed(self, seed: int):
        self._update("seed", seed)

    @property
    def learning_rate(self) -> float:
        return self.params["learning_rate"]

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        self._update("learning_rate", learning_rate)

    @property
    def optimizer(self) -> Optimizer:
        return self.params["optimizer"]

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer):
        self._update("optimizer", optimizer)

    @property
    def initializer_const(self) -> float:
        return self.params["initializer_const"]

    @initializer_const.setter
    def initializer_const(self, value: float):
        self._update("initializer_const", value)

    @property
    def initializer_uniform_scale(self) -> float:
        return self.params["initializer_uniform_scale"]

    @initializer_uniform_scale.setter
    def initializer_uniform_scale(self, value: float):
        self._update("initializer_uniform_scale", value)

    @property
    def error_function(self) -> ErrorFunction:
        return self.params["error_function"]

    @error_function.setter
    def error_function(self, error_function: ErrorFunction):
        self._update("error_function", error_function)

    @property
    def epochs(self) -> int:
        return self.params["epochs"]

    @epochs.setter
    def epochs(self, epochs: int):
        self._update("epochs", epochs)

    @property
    def initializer(self) -> Initializer:
        return self.params["initializer"]

    @initializer.setter
    def initializer(self, initializer: Initializer):
        self._update("initializer", initializer)

    @property
    def relation_neuron_activation(self) -> Activation:
        return self.params["relation_neuron_activation"]

    @relation_neuron_activation.setter
    def relation_neuron_activation(self, value: Activation):
        self._update("relation_neuron_activation", value)

    @property
    def rule_neuron_activation(self) -> Activation:
        return self.params["rule_neuron_activation"]

    @rule_neuron_activation.setter
    def rule_neuron_activation(self, value: Activation):
        self._update("rule_neuron_activation", value)

    def create_proxy(self) -> SettingsProxy:
        proxy = SettingsProxy(**self.params)
        self._proxies.add(proxy)

        return proxy

    def create_disconnected_proxy(self) -> SettingsProxy:
        return SettingsProxy(**self.params)

    def _update(self, parameter: str, value: Any) -> None:
        if parameter not in self.params:
            raise NotImplementedError
        self.params[parameter] = value

        for proxy in self._proxies.copy():
            proxy.__setattr__(parameter, value)
