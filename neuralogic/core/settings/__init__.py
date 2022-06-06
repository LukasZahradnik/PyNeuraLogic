from typing import Any, Optional
import weakref

from neuralogic.nn.init import Initializer, Uniform
from neuralogic.nn.loss import MSE, ErrorFunction
from neuralogic.core.settings.settings_proxy import SettingsProxy
from neuralogic.core.enums import Optimizer, Activation


class Settings:
    def __init__(
        self,
        *,
        optimizer: Optimizer = Optimizer.ADAM,
        learning_rate: Optional[float] = None,
        epochs: int = 3000,
        error_function: ErrorFunction = MSE(),
        initializer: Initializer = Uniform(),
        rule_activation: Activation = Activation.TANH,
        relation_activation: Activation = Activation.TANH,
        iso_value_compression: bool = True,
        chain_pruning: bool = True,
    ):
        self.params = locals().copy()
        self.params.pop("self")
        self._proxies = weakref.WeakSet()

        if learning_rate is None:
            self.params["learning_rate"] = 0.1 if optimizer == Optimizer.SGD else 0.001

    @property
    def iso_value_compression(self) -> bool:
        return self.params["iso_value_compression"]

    @iso_value_compression.setter
    def iso_value_compression(self, iso_value_compression: bool):
        self._update("iso_value_compression", iso_value_compression)

    @property
    def chain_pruning(self) -> bool:
        return self.params["chain_pruning"]

    @chain_pruning.setter
    def chain_pruning(self, chain_pruning: bool):
        self._update("chain_pruning", chain_pruning)

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
    def relation_activation(self) -> Activation:
        return self.params["relation_activation"]

    @relation_activation.setter
    def relation_activation(self, value: Activation):
        self._update("relation_activation", value)

    @property
    def rule_activation(self) -> Activation:
        return self.params["rule_activation"]

    @rule_activation.setter
    def rule_activation(self, value: Activation):
        self._update("rule_activation", value)

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
