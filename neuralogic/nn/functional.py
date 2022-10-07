from typing import Union

from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.core.constructs.function import Transformation, Combination, Function, Aggregation


# Transformation


def sigmoid(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Transformation.SIGMOID(entity)


def tanh(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Transformation.TANH(entity)


def signum(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Transformation.SIGNUM(entity)


def relu(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Transformation.RELU(entity)


def leaky_relu(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Transformation.LEAKY_RELU(entity)


def lukasiewicz(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Transformation.LUKASIEWICZ(entity)


def exp(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Transformation.EXP(entity)


def sqrt(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Transformation.SQRT(entity)


def inverse(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Transformation.INVERSE(entity)


def reverse(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Transformation.REVERSE(entity)


def log(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Transformation.LOG(entity)


def identity(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Transformation.IDENTITY(entity)


def transp(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Transformation.TRANSP(entity)


def softmax(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Transformation.SOFTMAX(entity)


def sparsemax(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Transformation.SPARSEMAX(entity)


# Combination


def max_comb(entity: BaseRelation = None) -> Union[BaseRelation, Function]:
    return Combination.MAX(entity)


def min_comb(entity: BaseRelation = None) -> Union[BaseRelation, Function]:
    return Combination.MIN(entity)


def avg_comb(entity: BaseRelation = None) -> Union[BaseRelation, Function]:
    return Combination.AVG(entity)


def sum_comb(entity: BaseRelation = None) -> Union[BaseRelation, Function]:
    return Combination.SUM(entity)


def count_comb(entity: BaseRelation = None) -> Union[BaseRelation, Function]:
    return Combination.COUNT(entity)


def product_comb(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Combination.PRODUCT(entity)


def elproduct_comb(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Combination.ELPRODUCT(entity)


def sparsemax_comb(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Combination.SPARSEMAX(entity)


def crosssum_comb(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Combination.CROSSSUM(entity)


def concat_comb(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Combination.CONCAT(entity)


def cossim_comb(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Combination.COSSIM(entity)


# Aggregations


def max(entity: BaseRelation = None) -> Union[BaseRelation, Function]:
    return Aggregation.MAX(entity)


def min(entity: BaseRelation = None) -> Union[BaseRelation, Function]:
    return Aggregation.MIN(entity)


def avg(entity: BaseRelation = None) -> Union[BaseRelation, Function]:
    return Aggregation.AVG(entity)


def sum(entity: BaseRelation = None) -> Union[BaseRelation, Function]:
    return Aggregation.SUM(entity)


def count(entity: BaseRelation = None) -> Union[BaseRelation, Function]:
    return Aggregation.COUNT(entity)
