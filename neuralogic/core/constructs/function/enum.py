from neuralogic.core.constructs.function.concat import Concat, ConcatComb
from neuralogic.core.constructs.function.function import (
    TransformationFunction,
    CombinationFunction,
    AggregationFunction,
)
from neuralogic.core.constructs.function.reshape import Reshape
from neuralogic.core.constructs.function.slice import Slice
from neuralogic.core.constructs.function.softmax import Softmax


class Transformation:
    # Element wise
    SIGMOID: TransformationFunction = TransformationFunction("SIGMOID")
    TANH: TransformationFunction = TransformationFunction("TANH")
    SIGNUM: TransformationFunction = TransformationFunction("SIGNUM")
    RELU: TransformationFunction = TransformationFunction("RELU")
    LEAKY_RELU: TransformationFunction = TransformationFunction("LEAKYRELU")
    LUKASIEWICZ: TransformationFunction = TransformationFunction("LUKASIEWICZ")
    EXP: TransformationFunction = TransformationFunction("EXP")
    SQRT: TransformationFunction = TransformationFunction("SQRT")
    INVERSE: TransformationFunction = TransformationFunction("INVERSE")
    REVERSE: TransformationFunction = TransformationFunction("REVERSE")
    LOG: TransformationFunction = TransformationFunction("LOG")

    # Transformation
    IDENTITY: TransformationFunction = TransformationFunction("IDENTITY")
    TRANSP: TransformationFunction = TransformationFunction("TRANSPOSE")
    SOFTMAX: TransformationFunction = TransformationFunction("SOFTMAX")
    SPARSEMAX: TransformationFunction = TransformationFunction("SPARSEMAX")
    NORM: TransformationFunction = TransformationFunction("NORM")
    SLICE: Slice = Slice("SLICE")
    RESHAPE: Reshape = Reshape("RESHAPE")


class Combination:
    # Aggregation
    AVG: CombinationFunction = CombinationFunction("AVG")
    MAX: CombinationFunction = CombinationFunction("MAX")
    MIN: CombinationFunction = CombinationFunction("MIN")
    SUM: CombinationFunction = CombinationFunction("SUM")
    COUNT: CombinationFunction = CombinationFunction("COUNT")

    # Combination
    PRODUCT: CombinationFunction = CombinationFunction("PRODUCT")
    ELPRODUCT: CombinationFunction = CombinationFunction("ELPRODUCT")
    SOFTMAX: CombinationFunction = CombinationFunction("SOFTMAX")
    SPARSEMAX: CombinationFunction = CombinationFunction("SPARSEMAX")
    CROSSSUM: CombinationFunction = CombinationFunction("CROSSSUM")
    CONCAT: ConcatComb = ConcatComb("CONCAT")
    COSSIM: CombinationFunction = CombinationFunction("COSSIM")


Combination.SUM.operator = "+"
Combination.ELPRODUCT.operator = "*"
Combination.PRODUCT.operator = "@"


class Aggregation:
    AVG: AggregationFunction = AggregationFunction("AVG")
    MAX: AggregationFunction = AggregationFunction("MAX")
    MIN: AggregationFunction = AggregationFunction("MIN")
    SUM: AggregationFunction = AggregationFunction("SUM")
    COUNT: AggregationFunction = AggregationFunction("COUNT")
    CONCAT: Concat = Concat("CONCAT")
    SOFTMAX: Softmax = Softmax("SOFTMAX")
