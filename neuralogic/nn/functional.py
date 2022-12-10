from typing import Union, Tuple, Sequence

from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.core.constructs.function import Transformation, Combination, Function, Aggregation


dot_type = type(Ellipsis)


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


def norm(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Transformation.NORM(entity)


def slice(
    entity: Union[BaseRelation, Function] = None,
    *,
    rows: Union[type(Ellipsis), Tuple[int, int]] = ...,
    cols: Union[type(Ellipsis), Tuple[int, int]] = ...,
) -> Union[BaseRelation, Function]:
    r"""
    Slices a value into a new value that is created by taking values on specified rows and columns.

    Rows and Cols coordinates are specified either as ``...``, which means all rows/cols or by a tuple of two
    elements ``[from_index, to_index]``.

    Parameters
    ----------

    entity : Union[BaseRelation, Function]
        Relation to apply the function on. Default: ``None``
    rows : Union[type(Ellipsis), Tuple[int, int]]
        Default: ``...``
    cols : Union[type(Ellipsis), Tuple[int, int]]
        Default: ``...``
    """
    return Transformation.SLICE(entity, rows=rows, cols=cols)


def reshape(
    entity: Union[BaseRelation, Function] = None,
    *,
    shape: Union[None, Tuple[int, int], int],
) -> Union[BaseRelation, Function]:
    r"""
    Change the shape/type of the value to a new shape. The shape can be either ``None``, int, or a tuple of two ints.

    * If ``None``, the underlying value will be converted to a scalar. E.g., a matrix value of one element ``[[1]]``
      will be converted to scalar ``1``.

    * If int, then the value will be converted to scalar (if the int is ``0``) or to a column vector.

    * If a tuple of two ints, the value will be converted to a scalar if the tuple is ``(0, 0)``. Into a row vector
      if the shape is ``(len, 0)`` or a column vector for shape ``(0, len)``. For other tuples ``(n, m)``,
      the value will be reshaped to matrix :math:`n \times m`.

    Parameters
    ----------

    entity : Union[BaseRelation, Function]
        Relation to apply the function on. Default: ``None``
    shape : Union[None, Tuple[int, int], int]
        The new shape of the value
    """
    return Transformation.RESHAPE(entity, shape=shape)


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


def softmax_comb(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Combination.SOFTMAX(entity)


def sparsemax_comb(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Combination.SPARSEMAX(entity)


def crosssum_comb(entity: Union[BaseRelation, Function] = None) -> Union[BaseRelation, Function]:
    return Combination.CROSSSUM(entity)


def concat_comb(entity: Union[BaseRelation, Function] = None, *, axis: int = -1) -> Union[BaseRelation, Function]:
    return Combination.CONCAT(entity, axis=axis)


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


def concat(entity: BaseRelation = None, *, axis: int = -1) -> Union[BaseRelation, Function]:
    return Aggregation.CONCAT(entity, axis=axis)


def softmax_agg(entity: BaseRelation = None, *, agg_terms: Sequence[str] = None) -> Union[BaseRelation, Function]:
    return Aggregation.SOFTMAX(entity, agg_terms=agg_terms)
