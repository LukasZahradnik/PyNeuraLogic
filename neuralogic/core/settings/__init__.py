from typing import Any, Optional
import weakref

from neuralogic.core.enums import Grounder
from neuralogic.nn.init import Initializer, Uniform
from neuralogic.nn.loss import MSE, ErrorFunction
from neuralogic.core.settings.settings_proxy import SettingsProxy
from neuralogic.core.constructs.function import Transformation, Combination, Aggregation
from neuralogic.optim import Optimizer, Adam


class Settings:
    def __init__(
        self,
        *,
        optimizer: Optimizer = Adam(),
        learning_rate: Optional[float] = None,
        epochs: int = 3000,
        error_function: ErrorFunction = MSE(),
        initializer: Initializer = Uniform(),
        rule_transformation: Transformation = Transformation.TANH,
        rule_combination: Combination = Combination.SUM,
        rule_aggregation: Aggregation = Aggregation.AVG,
        relation_transformation: Transformation = Transformation.TANH,
        relation_combination: Combination = Combination.SUM,
        iso_value_compression: bool = True,
        chain_pruning: bool = True,
        prune_only_identities: bool = False,
        grounder: Grounder = Grounder.BUP,
    ):
        self.params = locals().copy()
        self.params.pop("self")
        self._proxies: weakref.WeakSet[SettingsProxy] = weakref.WeakSet()

        self.kw_params = {}

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
    def prune_only_identities(self) -> bool:
        return self.params["prune_only_identities"]

    @prune_only_identities.setter
    def prune_only_identities(self, prune_only_identities: bool):
        self._update("prune_only_identities", prune_only_identities)

    @property
    def grounder(self) -> Grounder:
        return self.params["grounder"]

    @grounder.setter
    def grounder(self, grounder: Grounder):
        self._update("grounder", grounder)

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
    def relation_transformation(self) -> Transformation:
        return self.params["relation_transformation"]

    @relation_transformation.setter
    def relation_transformation(self, value: Transformation):
        self._update("relation_transformation", value)

    @property
    def relation_combination(self) -> Combination:
        return self.params["relation_combination"]

    @relation_combination.setter
    def relation_combination(self, value: Combination):
        self._update("relation_combination", value)

    @property
    def rule_transformation(self) -> Transformation:
        return self.params["rule_transformation"]

    @rule_transformation.setter
    def rule_transformation(self, value: Transformation):
        self._update("rule_transformation", value)

    @property
    def rule_combination(self) -> Combination:
        return self.params["rule_combination"]

    @rule_combination.setter
    def rule_combination(self, value: Combination):
        self._update("rule_combination", value)

    @property
    def rule_aggregation(self) -> Aggregation:
        return self.params["rule_aggregation"]

    @rule_aggregation.setter
    def rule_aggregation(self, value: Aggregation):
        self._update("rule_aggregation", value)

    def create_proxy(self) -> SettingsProxy:
        proxy = SettingsProxy(**self.params)
        self._proxies.add(proxy)

        for k, v in self.kw_params.items():
            proxy[k] = v

        return proxy

    def create_disconnected_proxy(self) -> SettingsProxy:
        proxy = SettingsProxy(**self.params)
        for k, v in self.kw_params.items():
            proxy[k] = v
        return proxy

    def __setitem__(self, key, value):
        for proxy in self._proxies.copy():
            proxy[key] = value
        self.kw_params[key] = value

    def __getitem__(self, item):
        return self.kw_params[item]

    def _update(self, parameter: str, value: Any) -> None:
        if parameter not in self.params:
            raise NotImplementedError
        self.params[parameter] = value

        for proxy in self._proxies.copy():
            proxy.__setattr__(parameter, value)
