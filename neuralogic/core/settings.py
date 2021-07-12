from neuralogic import get_neuralogic, get_gateway
from py4j.java_gateway import set_field, get_field
from enum import Enum


class Optimizer(str, Enum):
    ADAM = "ADAM"
    SGD = "SGD"


class ErrorFunction(str, Enum):
    SQUARED_DIFF = "SQUARED_DIFF"
    # ABS_DIFF = "ABS_DIFF"
    CROSSENTROPY = "CROSSENTROPY"


class Activation(Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SIGNUM = "signum"
    RELU = "relu"
    IDENTITY = "identity"
    LUKASIEWICZ = "lukasiewicz"
    SOFTMAX = "softmax"
    SPARSEMAX = "sparsemax"


class Aggregation(Enum):
    SUM = "sum"
    MAX = "max"
    AVG = "avg"


class Settings:
    def __init__(
        self,
        learning_rate: float = None,
        optimizer: Optimizer = None,
        epochs: int = None,
        error_function: ErrorFunction = None,
    ):
        self.namespace = get_neuralogic().cz.cvut.fel.ida.setup
        self.settings = self.namespace.Settings()

        if learning_rate is not None:
            self.learning_rate = learning_rate

        if optimizer is not None:
            self.optimizer = optimizer

        if epochs is not None:
            self.epochs = epochs

        self.error_function = ErrorFunction.SQUARED_DIFF if error_function is None else error_function
        set_field(self.settings, "debugExporting", False)
        set_field(self.settings, "isoValueCompression", False)  #todo gusta: tohle ja pak obecne zadouci mit zapnute (az na debugging)
        set_field(self.settings, "exportBlocks", get_gateway().new_array(get_gateway().jvm.java.lang.String, 0))
        self.settings.infer()

    @property
    def seed(self) -> int:
        return get_field(self.settings, "seed")

    @seed.setter
    def seed(self, seed: int):
        set_field(self.settings, "seed", seed)

    @property
    def learning_rate(self) -> float:
        return get_field(self.settings, "initLearningRate")

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        set_field(self.settings, "initLearningRate", learning_rate)

    @property
    def optimizer(self):
        return self.settings.getOptimizer()

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer):
        if optimizer == Optimizer.SGD:
            java_optimizer = self.namespace.Settings.OptimizerSet.SGD
        elif optimizer == Optimizer.ADAM:
            java_optimizer = self.namespace.Settings.OptimizerSet.ADAM
        else:
            raise NotImplementedError
        self.settings.setOptimizer(java_optimizer)

    @property
    def error_function(self):
        return self.settings.errorFunction

    @error_function.setter
    def error_function(self, error_function: ErrorFunction):
        if error_function == ErrorFunction.SQUARED_DIFF:
            java_error_function = self.namespace.Settings.ErrorFcn.SQUARED_DIFF
        # elif error_function == ErrorFunction.ABS_DIFF:
        #     java_error_function = self.namespace.Settings.ErrorFcn.ABS_DIFF
        elif error_function == ErrorFunction.CROSSENTROPY:
            java_error_function = self.namespace.Settings.ErrorFcn.CROSSENTROPY
        else:
            raise NotImplementedError
        self.settings.errorFunction = java_error_function

    @property
    def epochs(self) -> int:
        return get_field(self.settings, "maxCumEpochCount")

    @epochs.setter
    def epochs(self, epochs: int):
        set_field(self.settings, "maxCumEpochCount", epochs)

    @property
    def debug_exporting(self) -> bool:
        return get_field(self.settings, "debugExporting")

    @debug_exporting.setter
    def debug_exporting(self, debug_export: bool):
        set_field(self.settings, "debugExporting", debug_export)

    @property
    def default_fact_value(self) -> float:
        return get_field(self.settings, "defaultFactValue")

    @default_fact_value.setter
    def default_fact_value(self, value: float):
        set_field(self.settings, "defaultFactValue", value)

    def to_json(self) -> str:
        return self.settings.exportToJson()
