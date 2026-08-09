from typing import Any, Dict


class InitializerNames:
    UNIFORM = "UNIFORM"
    NORMAL = "NORMAL"
    CONSTANT = "CONSTANT"
    LONGTAIL = "LONGTAIL"
    GLOROT = "GLOROT"
    HE = "HE"
    TORCH = "TORCH"


class Initializer:
    def is_simple(self) -> bool:
        return True

    def get_settings(self) -> Dict[str, Any]:
        return {"initializer": str(self)}


class Uniform(Initializer):
    r"""Initializes learnable parameters with random uniformly distributed samples from the interval
    ``[-scale / 2, scale / 2]``.

    Parameters
    ----------

    scale : float
        Scale of the distribution interval ``[-scale / 2, scale / 2]``. Default: ``2``

    """

    def __init__(self, scale: float = 2):
        self.scale = scale

    def get_settings(self) -> Dict[str, Any]:
        return {
            "initializer": str(self),
            "initializer_uniform_scale": self.scale,
        }

    def __str__(self):
        return InitializerNames.UNIFORM


class Normal(Initializer):
    r"""Initializes learnable parameters with random samples from a normal (Gaussian) distribution"""

    def __str__(self):
        return InitializerNames.NORMAL


class Constant(Initializer):
    r"""Initializes learnable parameters with the ``value``.

    Parameters
    ----------

    value : float
        Value to fill weights with. Default: ``0.1``

    """

    def __init__(self, value: float = 0.1):
        self.value = value

    def get_settings(self) -> Dict[str, Any]:
        return {
            "initializer": str(self),
            "initializer_const": self.value,
        }

    def __str__(self):
        return InitializerNames.CONSTANT


class Longtail(Initializer):
    """Initializes learnable parameters with random samples from a long tail distribution"""

    def __str__(self):
        return InitializerNames.LONGTAIL


class Glorot(Initializer):
    r"""Initializes learnable parameters with samples from a uniform distribution (from the interval
    ``[-scale / 2, scale / 2]``) using the Glorot method.

    Parameters
    ----------

    scale : float
        Scale of a uniform distribution interval ``[-scale / 2, scale / 2]``. Default: ``2``

    """

    def __init__(self, scale: float = 2):
        self.scale = scale

    def is_simple(self) -> bool:
        return False

    def get_settings(self) -> Dict[str, Any]:
        return {
            "initializer": str(self),
            "initializer_uniform_scale": self.scale,
        }

    def __str__(self):
        return InitializerNames.GLOROT


class He(Initializer):
    r"""Initializes learnable parameters with samples from a uniform distribution (from the interval
    ``[-scale / 2, scale / 2]``) using the He method.

    Parameters
    ----------

    scale : float
        Scale of a uniform distribution interval ``[-scale / 2, scale / 2]``. Default: ``2``

    """

    def __init__(self, scale: float = 2):
        self.scale = scale

    def is_simple(self) -> bool:
        return False

    def get_settings(self) -> Dict[str, Any]:
        return {
            "initializer": str(self),
            "initializer_uniform_scale": self.scale,
        }

    def __str__(self):
        return InitializerNames.HE


class Torch(Initializer):
    r"""Initializes learnable parameters uniformly from ``[-1 / sqrt(fan_in), 1 / sqrt(fan_in)]``, which is
    what every layer PyTorch ships draws from.

    ``torch.nn.Linear`` uses ``kaiming_uniform_(a=sqrt(5))``, whose bound works out at exactly
    ``1 / sqrt(fan_in)``, and ``torch.nn.RNN``, ``LSTM`` and ``GRU`` draw every weight from
    ``U(+-1 / sqrt(hidden_size))``, which is the same for their square recurrent weight. Matching it means a
    model written here and the same model written there also start from the same place.

    A matrix takes its ``fan_in`` from its columns, which are the inputs it consumes. A vector keeps only one
    of its two declared dimensions, so a weight declared ``(1, n)`` consumes all n and is drawn from
    ``1 / sqrt(n)``, while one declared ``(n, 1)`` - or as a plain ``(n)`` - consumes one and keeps the full
    ``[-1, 1]``, exactly as ``torch.nn.Linear(1, n)`` does. A scalar likewise keeps ``[-1, 1]``.

    So only the weights whose fan-in is both unambiguous and large differ from :class:`~neuralogic.nn.init.Uniform`
    at all, which are the ones a wide template saturates on.

    Not the default, and not for anything it fails at. :class:`~neuralogic.nn.init.Glorot` is, because
    ``sqrt(6/(fan_in+fan_out))`` comes to ``sqrt(3/fan_in)`` for a square weight, the constant that actually
    keeps a uniformly drawn layer's output variance equal to its input's - where ``1/sqrt(fan_in)`` is
    ``sqrt(3)`` under it, ``0.125`` against ``0.2165`` on a 64x64. Torch's own source points at pytorch issue
    57109 about that. Reach for this one when a model here has to begin from the same distribution as the
    same model written in torch; the activation correction applies to either.

    Unlike :class:`~neuralogic.nn.init.Uniform`, :class:`~neuralogic.nn.init.Glorot` and
    :class:`~neuralogic.nn.init.He`, this takes no scale - the whole point is that the spread follows the
    shape of the weight rather than a number chosen in advance.
    """

    def is_simple(self) -> bool:
        return False

    def __str__(self):
        return InitializerNames.TORCH


__all__ = ["Normal", "Uniform", "Constant", "Longtail", "Glorot", "He", "Torch", "Initializer", "InitializerNames"]
