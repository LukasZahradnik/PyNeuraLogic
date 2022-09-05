class Function:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name: str = name.lower()

    def __str__(self):
        return self.name

    def pretty_str(self) -> str:
        return str(self).capitalize()

    def __call__(self, *args):
        if len(args) == 0 or args[0] is None:
            return self
        raise NotImplementedError


class Transformation(Function):
    # Element wise
    SIGMOID: "Transformation"
    TANH: "Transformation"
    SIGNUM: "Transformation"
    RELU: "Transformation"
    LEAKY_RELU: "Transformation"
    LUKASIEWICZ: "Transformation"
    EXP: "Transformation"
    SQRT: "Transformation"
    INVERSE: "Transformation"
    REVERSE: "Transformation"
    LOG: "Transformation"

    # Transformation
    IDENTITY: "Transformation"
    TRANSP: "Transformation"
    SOFTMAX: "Transformation"
    SPARSEMAX: "Transformation"

    def __call__(self, *args):
        from neuralogic.core.constructs import relation

        if len(args) == 0 or args[0] is None:
            return self

        arg = args[0]
        if isinstance(arg, relation.BaseRelation):
            return arg.attach_activation_function(self)
        raise NotImplementedError


_special_namings = {"LEAKY_RELU": "LEAKYRELU", "TRANSP": "TRANSPOSE"}

for function_name in Transformation.__annotations__:
    setattr(Transformation, function_name, Transformation(_special_namings.get(function_name, function_name)))


class Combination(Function):
    # Aggregation
    AVG: "Combination"
    MAX: "Combination"
    MIN: "Combination"
    SUM: "Combination"
    COUNT: "Combination"

    # Combination
    PRODUCT: "Combination"
    ELPRODUCT: "Combination"
    SOFTMAX: "Combination"
    SPARSEMAX: "Combination"
    CROSSSUM: "Combination"
    CONCAT: "Combination"
    COSSIM: "Combination"


for function_name in Combination.__annotations__:
    setattr(Combination, function_name, Combination(function_name))


class Aggregation(Function):
    AVG: "Aggregation"
    MAX: "Aggregation"
    MIN: "Aggregation"
    SUM: "Aggregation"
    COUNT: "Aggregation"


for function_name in Aggregation.__annotations__:
    setattr(Aggregation, function_name, Aggregation(function_name))
