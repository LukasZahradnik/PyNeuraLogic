from enum import Enum
from dataclasses import dataclass


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


class Metadata:
    def __init__(
        self, offset=None, learnable: bool = None, activation: Activation = None, aggregation: Aggregation = None
    ):
        self.offset = offset
        self.learnable = learnable
        self.activation = activation
        self.aggregation = aggregation

    def __str__(self):
        metadata_list = []
        if self.offset is not None:
            metadata_list.append(f"offset={self.offset}")
        if self.learnable is not None:
            metadata_list.append(f"learnable={str(self.learnable).lower()}")
        if self.activation is not None:
            metadata_list.append(f"activation={self.activation.value}")
        if self.aggregation is not None:
            metadata_list.append(f"aggregation={self.aggregation.value}")
        return f"[{', '.join(metadata_list)}]"
