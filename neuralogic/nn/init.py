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
        raise {"initializer": str(self)}


class Uniform(Initializer):
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
    def __str__(self):
        return InitializerNames.NORMAL


class Constant(Initializer):
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
    def __str__(self):
        return InitializerNames.LONGTAIL


class Glorot(Initializer):
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
