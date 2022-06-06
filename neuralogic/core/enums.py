from enum import Enum


class Optimizer(str, Enum):
    ADAM = "ADAM"
    SGD = "SGD"


class Activation(str, Enum):
    SIGMOID = "SIGMOID"
    TANH = "TANH"
    SIGNUM = "SIGNUM"
    RELU = "RELU"
    LEAKY_RELU = "LEAKYRELU"
    IDENTITY = "IDENTITY"
    LUKASIEWICZ = "LUKASIEWICZ"
    SOFTMAX = "SOFTMAX"
    SPARSEMAX = "SPARSEMAX"

    def __add__(self, other: "ActivationAgg"):
        if not isinstance(other, ActivationAgg):
            raise NotImplementedError
        return ActivationAggregation(other, self)

    def __str__(self):
        return self.value.lower()


class ActivationAgg(str, Enum):
    MAX = "max"
    MIN = "min"

    def __add__(self, other: Activation):
        if not isinstance(other, Activation):
            raise NotImplementedError
        return ActivationAggregation(self, other)

    def __str__(self):
        return f"{self.value}-identity"


class Aggregation(str, Enum):
    SUM = "sum"
    MAX = "max"
    AVG = "avg"
    MIN = "min"


class Backend(Enum):
    DYNET = "dynet"
    JAVA = "java"
    TORCH = "torch"


class ActivationAggregation:
    def __init__(self, aggregation: ActivationAgg, activation: Activation):
        self.aggregation = aggregation
        self.activation = activation

    def __str__(self):
        return f"{self.aggregation.value}-{self.activation.value.lower()}"
