from neuralogic.core.constructs.function.concat import ConcatComb, Concat
from neuralogic.core.constructs.function.function import Transformation, Combination, Aggregation, Function
from neuralogic.core.constructs.function.reshape import Reshape
from neuralogic.core.constructs.function.slice import Slice
from neuralogic.core.constructs.function.softmax import Softmax

_special_namings = {"LEAKY_RELU": "LEAKYRELU", "TRANSP": "TRANSPOSE"}

for function_name in Transformation.__annotations__:
    setattr(Transformation, function_name, Transformation(_special_namings.get(function_name, function_name)))


Transformation.SLICE = Slice("SLICE")
Transformation.RESHAPE = Reshape("RESHAPE")


for function_name in Combination.__annotations__:
    setattr(Combination, function_name, Combination(function_name))


Combination.CONCAT = ConcatComb("CONCAT")


for function_name in Aggregation.__annotations__:
    setattr(Aggregation, function_name, Aggregation(function_name))


Aggregation.CONCAT = Concat("CONCAT")
Aggregation.SOFTMAX = Softmax("SOFTMAX")


__all__ = ["Transformation", "Combination", "Aggregation", "Function"]
