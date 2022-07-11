from typing import Callable


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


class Activation(Function):
    LUKASIEWICZ: "Activation" = None
    SIGMOID: "Activation" = None
    SIGNUM: "Activation" = None
    RELU: "Activation" = None
    LEAKY_RELU: "Activation" = None
    IDENTITY: "Activation" = None
    TANH: "Activation" = None
    EXP: "Activation" = None
    TRANSP: "Activation" = None
    NORM: "Activation" = None
    SQRT: "Activation" = None
    INVERSE: "Activation" = None
    REVERSE: "Activation" = None
    SOFTMAX: "Activation" = None
    SPARSEMAX: "Activation" = None

    __slots__ = ("nestable",)

    def __init__(self, name: str):
        super().__init__(name)
        self.nestable = True

    def nest(self, other: "Function") -> "Function":
        if isinstance(other, Aggregation):
            raise NotImplementedError(f"Cannot nest aggregation functions ({self} with {other})")

        if isinstance(other, Activation) and isinstance(self, Activation):
            raise NotImplementedError(f"Cannot nest activation functions ({self} with {other})")

        if not self.nestable:
            raise NotImplementedError(f"Cannot nest nested functions - {self} and {other}")

        function = Activation(f"{other.name}-{self.name}")
        function.nestable = False

        return function

    def __call__(self, *args):
        from neuralogic.core.constructs import relation

        if len(args) == 0 or args[0] is None:
            return self

        arg = args[0]

        if isinstance(arg, Callable) and not isinstance(arg, (ActivationAgg, relation.BaseRelation)):
            arg = arg()
        if isinstance(arg, relation.BaseRelation):
            return arg.attach_activation_function(self)
        if isinstance(arg, ActivationAgg):
            return self.nest(arg)
        raise NotImplementedError


Activation.LUKASIEWICZ = Activation("LUKASIEWICZ")
Activation.SIGMOID = Activation("SIGMOID")
Activation.SIGNUM = Activation("SIGNUM")
Activation.RELU = Activation("RELU")
Activation.LEAKY_RELU = Activation("LEAKYRELU")
Activation.IDENTITY = Activation("IDENTITY")
Activation.TANH = Activation("TANH")
Activation.EXP = Activation("EXP")
Activation.TRANSP = Activation("TRANSPOSE")
Activation.NORM = Activation("NORM")
Activation.SQRT = Activation("SQRT")
Activation.INVERSE = Activation("INVERSE")
Activation.REVERSE = Activation("REVERSE")
Activation.SOFTMAX = Activation("SOFTMAX")
Activation.SPARSEMAX = Activation("SPARSEMAX")


class ActivationAgg(Function):
    CROSSUM: "ActivationAgg" = None
    ELEMENTPRODUCT: "ActivationAgg" = None
    PRODUCT: "ActivationAgg" = None
    CONCAT: "ActivationAgg" = None
    MAX: "ActivationAgg" = None
    MIN: "ActivationAgg" = None

    def nest(self, other: "Function") -> "Function":
        if isinstance(other, Activation):
            return other.nest(self)

        if isinstance(other, ActivationAgg):
            raise NotImplementedError(f"Cannot nest act-aggregation functions ({self} with {other}")
        raise NotImplementedError

    def __str__(self):
        return f"{self.name}-identity"

    def __call__(self, *args):
        from neuralogic.core.constructs import relation

        if len(args) == 0 or args[0] is None:
            return self

        arg = args[0]

        if isinstance(arg, Callable) and not isinstance(arg, (Activation, relation.BaseRelation)):
            arg = arg()
        if isinstance(arg, relation.BaseRelation):
            return arg.attach_activation_function(self)
        if isinstance(arg, Activation):
            return self.nest(arg)
        raise NotImplementedError


ActivationAgg.CROSSUM = ActivationAgg("CROSSUM")
ActivationAgg.ELEMENTPRODUCT = ActivationAgg("ELEMENTPRODUCT")
ActivationAgg.PRODUCT = ActivationAgg("PRODUCT")
ActivationAgg.CONCAT = ActivationAgg("CONCAT")
ActivationAgg.MAX = ActivationAgg("MAX")
ActivationAgg.MIN = ActivationAgg("MIN")


class Aggregation(Function):
    MAX: "Aggregation" = None
    MIN: "Aggregation" = None
    AVG: "Aggregation" = None
    SUM: "Aggregation" = None


Aggregation.MAX = Aggregation("MAX")
Aggregation.MIN = Aggregation("MIN")
Aggregation.AVG = Aggregation("AVG")
Aggregation.SUM = Aggregation("SUM")
