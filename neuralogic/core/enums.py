from enum import Enum


class Optimizer(str, Enum):
    ADAM = "ADAM"
    SGD = "SGD"


class ErrorFunction(str, Enum):
    SQUARED_DIFF = "SQUARED_DIFF"
    # ABS_DIFF = "ABS_DIFF"
    CROSSENTROPY = "CROSSENTROPY"
    SOFTENTROPY = "SOFTENTROPY"


class Activation(Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SIGNUM = "signum"
    RELU = "relu"
    IDENTITY = "identity"
    LUKASIEWICZ = "lukasiewicz"
    SOFTMAX = "softmax"
    SPARSEMAX = "sparsemax"


class Aggregation(Enum):
    SUM = "sum"
    MAX = "max"
    AVG = "avg"


class Initializer(Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    CONSTANT = "constant"
    LONGTAIL = "longtail"
    GLOROT = "glorot"
    HE = "he"


class Backend(Enum):
    DYNET = "dynet"
    PYG = "pyg"
    DGL = "dgl"
    JAVA = "java"
