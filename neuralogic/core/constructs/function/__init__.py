from neuralogic.core.constructs.function.function import Transformation, Combination, Aggregation, Function
from neuralogic.core.constructs.function.slice import Slice

_special_namings = {"LEAKY_RELU": "LEAKYRELU", "TRANSP": "TRANSPOSE"}

for function_name in Transformation.__annotations__:
    setattr(Transformation, function_name, Transformation(_special_namings.get(function_name, function_name)))


Transformation.SLICE = Slice("SLICE")


for function_name in Combination.__annotations__:
    setattr(Combination, function_name, Combination(function_name))


for function_name in Aggregation.__annotations__:
    setattr(Aggregation, function_name, Aggregation(function_name))


__all__ = ["Transformation", "Combination", "Aggregation", "Function"]
