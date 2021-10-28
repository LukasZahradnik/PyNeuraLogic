from enum import Enum


class Optimizer(str, Enum):
    ADAM = "ADAM"
    SGD = "SGD"


class ErrorFunction(str, Enum):
    SQUARED_DIFF = "SQUARED_DIFF"
    # ABS_DIFF = "ABS_DIFF"
    CROSSENTROPY = "CROSSENTROPY"
    SOFTENTROPY = "SOFTENTROPY"


class Activation(str, Enum):
    SIGMOID = "SIGMOID"
    TANH = "TANH"
    SIGNUM = "SIGNUM"
    RELU = "RELU"
    IDENTITY = "IDENTITY"
    LUKASIEWICZ = "LUKASIEWICZ"
    SOFTMAX = "SOFTMAX"
    SPARSEMAX = "SPARSEMAX"


class Aggregation(Enum):
    SUM = "sum"
    MAX = "max"
    AVG = "avg"


class Initializer(str, Enum):
    UNIFORM = "UNIFORM"
    NORMAL = "NORMAL"
    CONSTANT = "CONSTANT"
    LONGTAIL = "LONGTAIL"
    GLOROT = "GLOROT"
    HE = "HE"


class Backend(Enum):
    DYNET = "dynet"
    JAVA = "java"
    PYG = "pyg"
    TORCH = "torch"
