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
    SIGMOID: "Transformation" = None
    TANH: "Transformation" = None
    SIGNUM: "Transformation" = None
    RELU: "Transformation" = None
    LEAKY_RELU: "Transformation" = None
    LUKASIEWICZ: "Transformation" = None
    EXP: "Transformation" = None
    SQRT: "Transformation" = None
    INVERSE: "Transformation" = None
    REVERSE: "Transformation" = None

    # Transformation
    IDENTITY: "Transformation" = None
    TRANSP: "Transformation" = None
    SOFTMAX: "Transformation" = None
    SPARSEMAX: "Transformation" = None

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
    AVG: "Combination" = None
    MAX: "Combination" = None
    MIN: "Combination" = None
    SUM: "Combination" = None
    COUNT: "Combination" = None

    # Combination
    PRODUCT: "Combination" = None
    ELPRODUCT: "Combination" = None
    SOFTMAX: "Combination" = None
    SPARSEMAX: "Combination" = None
    CROSSSUM: "Combination" = None
    CONCAT: "Combination" = None
    COSSIM: "Combination" = None


for function_name in Combination.__annotations__:
    setattr(Combination, function_name, Combination(function_name))


class Aggregation(Function):
    AVG: "Aggregation" = None
    MAX: "Aggregation" = None
    MIN: "Aggregation" = None
    SUM: "Aggregation" = None
    COUNT: "Aggregation" = None


for function_name in Aggregation.__annotations__:
    setattr(Aggregation, function_name, Aggregation(function_name))
