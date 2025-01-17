from neuralogic.core.constructs.function.concat import ConcatAggregation, ConcatCombination
from neuralogic.core.constructs.function.function import (
    TransformationFunction,
    CombinationFunction,
    AggregationFunction,
)
from neuralogic.core.constructs.function.reshape import Reshape
from neuralogic.core.constructs.function.slice import Slice
from neuralogic.core.constructs.function.softmax import SoftmaxAggregation


class Transformation:
    # Element wise
    SIGMOID: TransformationFunction = TransformationFunction("SIGMOID")
    TANH: TransformationFunction = TransformationFunction("TANH")
    SIGNUM: TransformationFunction = TransformationFunction("SIGNUM")
    RELU: TransformationFunction = TransformationFunction("RELU", namespace="transformation.elementwise.ReLu")
    LEAKY_RELU: TransformationFunction = TransformationFunction(
        "LEAKYRELU", namespace="transformation.elementwise.LeakyReLu"
    )
    LUKASIEWICZ: TransformationFunction = TransformationFunction(
        "LUKASIEWICZ", namespace="transformation.elementwise.LukasiewiczSigmoid"
    )
    EXP: TransformationFunction = TransformationFunction("EXP", namespace="transformation.elementwise.Exponentiation")
    SQRT: TransformationFunction = TransformationFunction("SQRT", namespace="transformation.elementwise.SquareRoot")
    INVERSE: TransformationFunction = TransformationFunction("INVERSE")
    REVERSE: TransformationFunction = TransformationFunction("REVERSE")
    LOG: TransformationFunction = TransformationFunction("LOG", namespace="transformation.elementwise.Logarithm")

    # Transformation
    IDENTITY: TransformationFunction = TransformationFunction("IDENTITY", namespace="transformation.join.{name}")
    TRANSP: TransformationFunction = TransformationFunction("TRANSPOSE", namespace="transformation.join.Transposition")
    SOFTMAX: TransformationFunction = TransformationFunction("SOFTMAX", namespace="transformation.join.{name}")
    SPARSEMAX: TransformationFunction = TransformationFunction("SPARSEMAX", namespace="transformation.join.{name}")
    NORM: TransformationFunction = TransformationFunction("NORM", namespace="transformation.join.Normalization")
    SLICE: Slice = Slice("SLICE")
    RESHAPE: Reshape = Reshape("RESHAPE")


class Combination:
    # Aggregation
    AVG: CombinationFunction = CombinationFunction("AVG", namespace="aggregation.Average")
    MAX: CombinationFunction = CombinationFunction("MAX", can_flatten=True, namespace="aggregation.Maximum")
    MIN: CombinationFunction = CombinationFunction("MIN", can_flatten=True, namespace="aggregation.Minimum")
    SUM: CombinationFunction = CombinationFunction(
        "SUM", operator="+", can_flatten=True, namespace="aggregation.{name}"
    )
    COUNT: CombinationFunction = CombinationFunction("COUNT", namespace="aggregation.{name}")

    # Combination
    PRODUCT: CombinationFunction = CombinationFunction("PRODUCT", operator="@")
    ELPRODUCT: CombinationFunction = CombinationFunction(
        "ELPRODUCT", operator="*", can_flatten=True, namespace="combination.ElementProduct"
    )
    SOFTMAX: CombinationFunction = CombinationFunction("SOFTMAX")
    SPARSEMAX: CombinationFunction = CombinationFunction("SPARSEMAX")
    CROSSSUM: CombinationFunction = CombinationFunction("CROSSSUM")
    CONCAT: ConcatCombination = ConcatCombination("CONCAT")
    COSSIM: CombinationFunction = CombinationFunction("COSSIM")


class Aggregation:
    AVG: AggregationFunction = AggregationFunction("AVG")
    MAX: AggregationFunction = AggregationFunction("MAX")
    MIN: AggregationFunction = AggregationFunction("MIN")
    SUM: AggregationFunction = AggregationFunction("SUM")
    COUNT: AggregationFunction = AggregationFunction("COUNT")
    CONCAT: ConcatAggregation = ConcatAggregation("CONCAT")
    SOFTMAX: SoftmaxAggregation = SoftmaxAggregation("SOFTMAX")


class F:
    # Element wise
    sigmoid: TransformationFunction = Transformation.SIGMOID
    tanh: TransformationFunction = Transformation.TANH
    signum: TransformationFunction = Transformation.SIGNUM
    relu: TransformationFunction = Transformation.RELU
    leaky_relu: TransformationFunction = Transformation.LEAKY_RELU
    lukasiewicz: TransformationFunction = Transformation.LUKASIEWICZ
    exp: TransformationFunction = Transformation.EXP
    sqrt: TransformationFunction = Transformation.SQRT
    inverse: TransformationFunction = Transformation.INVERSE
    reverse: TransformationFunction = Transformation.REVERSE
    log: TransformationFunction = Transformation.LOG

    # Transformation
    identity: TransformationFunction = Transformation.IDENTITY
    transp: TransformationFunction = Transformation.TRANSP
    softmax: TransformationFunction = Transformation.SOFTMAX
    sparsemax: TransformationFunction = Transformation.SPARSEMAX
    norm: TransformationFunction = Transformation.NORM
    slice: Slice = Transformation.SLICE
    reshape: Reshape = Transformation.RESHAPE

    # Combination
    avg: CombinationFunction = Combination.AVG
    max: CombinationFunction = Combination.MAX
    min: CombinationFunction = Combination.MIN
    sum: CombinationFunction = Combination.SUM
    count: CombinationFunction = Combination.COUNT

    product: CombinationFunction = Combination.PRODUCT
    elproduct: CombinationFunction = Combination.ELPRODUCT
    softmax_comb: CombinationFunction = Combination.SOFTMAX
    sparsemax_comb: CombinationFunction = Combination.SPARSEMAX
    crossum: CombinationFunction = Combination.CROSSSUM
    concat: ConcatCombination = Combination.CONCAT
    cossim: CombinationFunction = Combination.COSSIM

    # Aggregation
    avg_agg: AggregationFunction = Aggregation.AVG
    max_agg: AggregationFunction = Aggregation.MAX
    min_agg: AggregationFunction = Aggregation.MIN
    sum_agg: AggregationFunction = Aggregation.SUM
    count_agg: AggregationFunction = Aggregation.COUNT
    concat_agg: ConcatAggregation = Aggregation.CONCAT
    softmax_agg: SoftmaxAggregation = Aggregation.SOFTMAX
