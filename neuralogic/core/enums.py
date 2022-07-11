from enum import Enum


class Optimizer(str, Enum):
    ADAM = "ADAM"
    SGD = "SGD"


class Backend(Enum):
    DYNET = "dynet"
    JAVA = "java"
    TORCH = "torch"
