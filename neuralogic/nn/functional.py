from typing import Union

from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.core.constructs.function import Activation, ActivationAgg, Function, Aggregation


# Activations


def lukasiewicz(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Activation.LUKASIEWICZ(entity)


def sigmoid(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Activation.SIGMOID(entity)


def signum(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Activation.SIGNUM(entity)


def relu(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Activation.RELU(entity)


def leaky_relu(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Activation.LEAKY_RELU(entity)


def identity(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Activation.IDENTITY(entity)


def tanh(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Activation.TANH(entity)


def exp(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Activation.EXP(entity)


def transp(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Activation.TRANSP(entity)


def norm(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Activation.NORM(entity)


def sqrt(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Activation.SQRT(entity)


def inverse(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Activation.INVERSE(entity)


def reverse(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Activation.REVERSE(entity)


def softmax(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Activation.SOFTMAX(entity)


def sparsemax(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Activation.SPARSEMAX(entity)


# Activation-Aggregations


def crossum(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return ActivationAgg.CROSSUM(entity)


def elementproduct(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return ActivationAgg.ELEMENTPRODUCT(entity)


def product(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return ActivationAgg.PRODUCT(entity)


def concat(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return ActivationAgg.CONCAT(entity)


def max_act(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return ActivationAgg.MAX(entity)


def min_act(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return ActivationAgg.MIN(entity)


# Aggregations


def max(entity: BaseRelation = None) -> Union[BaseRelation, Function]:
    return Aggregation.MAX(entity)


def min(entity: BaseRelation = None) -> Union[BaseRelation, Function]:
    return Aggregation.MIN(entity)


def avg(entity: BaseRelation = None) -> Union[BaseRelation, Function]:
    return Aggregation.AVG(entity)


def sum(entity: BaseRelation = None) -> Union[BaseRelation, Function]:
    return Aggregation.SUM(entity)
