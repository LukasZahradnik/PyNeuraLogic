from neuralogic import get_neuralogic, get_gateway
from py4j.java_gateway import set_field, get_field
from enum import Enum


class Optimizer(str, Enum):
    ADAM = "ADAM"
    SGD = "SGD"


class Settings:
    def __init__(self, learning_rate: float = None, optimizer: Optimizer = None, epochs: int = None):
        self.namespace = get_neuralogic().cz.cvut.fel.ida.setup
        self.settings = self.namespace.Settings()

        if learning_rate is not None:
            self.learning_rate = learning_rate

        if optimizer is not None:
            self.optimizer = optimizer

        if epochs is not None:
            self.epochs = epochs

        set_field(self.settings, "debugExporting", False)
        set_field(self.settings, "isoValueCompression", False)
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
