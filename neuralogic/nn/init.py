from typing import Dict, Any


class InitializerNames:
    UNIFORM = "UNIFORM"
    NORMAL = "NORMAL"
    CONSTANT = "CONSTANT"
    LONGTAIL = "LONGTAIL"
    GLOROT = "GLOROT"
    HE = "HE"


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


__all__ = ["Normal", "Uniform", "Constant", "Longtail", "Glorot", "He", "Initializer", "InitializerNames"]
