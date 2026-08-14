import weakref
from typing import Any

from neuralogic.core.enums import Grounder
from neuralogic.core.settings.settings_proxy import SettingsProxy
from neuralogic.nn.init import Glorot, Initializer
from neuralogic.nn.loss import MSE, ErrorFunction
from neuralogic.nn.optim import Adam, Optimizer


class Settings:
    def __init__(
        self,
        *,
        optimizer: Optimizer = Adam(),
        error_function: ErrorFunction = MSE(),
        initializer: Initializer = Glorot(),
        iso_value_compression: bool = True,
        chain_pruning: bool = True,
        prune_only_identities: bool = False,
        grounder: Grounder = Grounder.BUP,
        clip_grad_norm: float | None = None,
        clip_grad_value: float | None = None,
    ):
        """Gradient clipping lives here rather than on the optimizer, which is where torch does *not* put it
        either: :code:`clip_grad_norm_` is a call made between :code:`backward()` and :code:`step()`. There is
        no such hook here - the engine owns the whole iteration - so the only place left to say it is the
        settings. Weight decay, which torch *does* put on the optimizer, is on the optimizer.
        """
        self.params = locals().copy()
        self.params.pop("self")
        self._proxies: weakref.WeakSet[SettingsProxy] = weakref.WeakSet()

        self.kw_params: dict[str, Any] = {}

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
    def clip_grad_norm(self) -> float | None:
        return self.params["clip_grad_norm"]

    @clip_grad_norm.setter
    def clip_grad_norm(self, clip_grad_norm: float | None):
        self._update("clip_grad_norm", clip_grad_norm)

    @property
    def clip_grad_value(self) -> float | None:
        return self.params["clip_grad_value"]

    @clip_grad_value.setter
    def clip_grad_value(self, clip_grad_value: float | None):
        self._update("clip_grad_value", clip_grad_value)

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
    def initializer(self) -> Initializer:
        return self.params["initializer"]

    @initializer.setter
    def initializer(self, initializer: Initializer):
        self._update("initializer", initializer)

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
