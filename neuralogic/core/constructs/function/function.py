from typing import Optional

import jpype


class Function:
    __slots__ = "name", "operator"

    def __init__(self, name: str):
        self.name: str = name.lower()
        self.operator: Optional[str] = None

    def __str__(self):
        return self.name

    def wrap(self, content: str) -> str:
        return f"{self.name}({content})"

    def pretty_str(self) -> str:
        return str(self).capitalize()

    def __call__(self, *args, **kwargs):
        if len(args) == 0 or args[0] is None:
            return self
        raise NotImplementedError

    def is_parametrized(self) -> bool:
        return False

    def get(self):
        raise NotImplementedError

    def rule_head_dependant(self) -> bool:
        return False

    def process_head(self, head) -> "Function":
        pass


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
    NORM: "Transformation"
    SLICE: "Transformation"
    RESHAPE: "Transformation"

    def get(self):
        name = "".join(s.capitalize() for s in self.name.split("_"))

        if name == "Transp":
            name = "Transposition"
        if name == "Norm":
            name = "Normalization"

        if name in ("Identity", "Transposition", "Softmax", "Sparsemax", "Normalization", "Slice", "Reshape"):
            return jpype.JClass(f"cz.cvut.fel.ida.algebra.functions.transformation.joint.{name}")()
        return jpype.JClass(f"cz.cvut.fel.ida.algebra.functions.transformation.elementwise.{name}")()

    def __call__(self, *args, **kwargs):
        from neuralogic.core.constructs import relation
        from neuralogic.core.constructs.function.function_container import FContainer

        if len(args) == 0 or args[0] is None:
            return self

        arg = args[0]
        if isinstance(arg, relation.BaseRelation):
            if arg.function is not None:
                return FContainer((arg,), self)
            return arg.attach_activation_function(self)
        return FContainer(args, self)


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

    def get(self):
        name = "".join(s.capitalize() for s in self.name.split("_"))

        if name in ("Sum", "Max", "Min", "Avg", "Count"):
            return jpype.JClass(f"cz.cvut.fel.ida.algebra.functions.aggregation.{name}")()
        if name == "Elproduct":
            name = "ElementProduct"

        return jpype.JClass(f"cz.cvut.fel.ida.algebra.functions.combination.{name}")()

    def __call__(self, *args, **kwargs):
        from neuralogic.core.constructs.function.function_container import FContainer

        if len(args) == 0 or args[0] is None:
            return self
        return FContainer(args, self)


class Aggregation(Function):
    AVG: "Aggregation"
    MAX: "Aggregation"
    MIN: "Aggregation"
    SUM: "Aggregation"
    COUNT: "Aggregation"
    CONCAT: "Aggregation"
    SOFTMAX: "Aggregation"
