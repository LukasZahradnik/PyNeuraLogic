from typing import Optional

from neuralogic import get_neuralogic, get_gateway
from neuralogic.core.enums import Optimizer, Initializer, ErrorFunction, Activation
from py4j.java_gateway import set_field, get_field


class SettingsProxy:
    def __init__(
            self,
            *,
            optimizer: Optimizer,
            learning_rate: float,
            epochs: Optional[int],
            error_function: ErrorFunction,
            initializer: Initializer,
            initializer_const: float,
            initializer_uniform_scale: float,
            rule_neuron_activation: Activation,
            relation_neuron_activation: Activation,
    ):
        self.namespace = get_neuralogic().cz.cvut.fel.ida.setup
        self.settings = self.namespace.Settings()

        params = locals().copy()
        params.pop("self")

        for key, value in params.items():
            if value is not None:
                self.__setattr__(key, value)

        set_field(self.settings, "debugExporting", False)
        set_field(self.settings, "isoValueCompression", True)
        set_field(self.settings, "squishLastLayer", False)
        set_field(self.settings, "trainOnlineResultsType", self.namespace.Settings.ResultsType.REGRESSION)
        set_field(self.settings, "exportBlocks", get_gateway().new_array(get_gateway().jvm.java.lang.String, 0))

        self.settings.infer()

    @property
    def iso_value_compression(self) -> bool:
        return get_field(self.settings, "isoValueCompression")

    @iso_value_compression.setter
    def iso_value_compression(self, iso_value_compression: bool):
        set_field(self.settings, "isoValueCompression", iso_value_compression)

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
    def initializer_const(self):
        return get_field(self.settings, "constantInitValue")

    @initializer_const.setter
    def initializer_const(self, value: float):
        set_field(self.settings, "constantInitValue", value)

    @property
    def initializer_uniform_scale(self):
        return get_field(self.settings, "randomInitScale")

    @initializer_uniform_scale.setter
    def initializer_uniform_scale(self, value: float):
        set_field(self.settings, "randomInitScale", value)

    @property
    def error_function(self):
        return get_field(self.settings, "errorFunction")

    @error_function.setter
    def error_function(self, error_function: ErrorFunction):
        if error_function == ErrorFunction.SQUARED_DIFF:
            java_error_function = self.namespace.Settings.ErrorFcn.SQUARED_DIFF
        # elif error_function == ErrorFunction.ABS_DIFF:
        #     java_error_function = self.namespace.Settings.ErrorFcn.ABS_DIFF
        elif error_function == ErrorFunction.SOFTENTROPY:
            java_error_function = self.namespace.Settings.ErrorFcn.SOFTENTROPY
        elif error_function == ErrorFunction.CROSSENTROPY:
            java_error_function = self.namespace.Settings.ErrorFcn.CROSSENTROPY
        else:
            raise NotImplementedError
        set_field(self.settings, "errorFunction", java_error_function)

    @property
    def epochs(self) -> int:
        return get_field(self.settings, "maxCumEpochCount")

    @epochs.setter
    def epochs(self, epochs: int):
        set_field(self.settings, "maxCumEpochCount", epochs)

    @property
    def initializer(self):
        initializer = get_field(self.settings, "initializer")

        if str(initializer) != "SIMPLE":
            return initializer
        return get_field(self.settings, "initDistribution")

    @initializer.setter
    def initializer(self, initializer: Initializer):
        if initializer == Initializer.HE:
            set_field(self.settings, "initializer", self.namespace.Settings.InitSet.HE)
            return
        if initializer == Initializer.GLOROT:
            set_field(self.settings, "initializer", self.namespace.Settings.InitSet.GLOROT)
            return

        if initializer == Initializer.NORMAL:
            init_dist = self.namespace.Settings.InitDistribution.NORMAL
        elif initializer == Initializer.UNIFORM:
            init_dist = self.namespace.Settings.InitDistribution.UNIFORM
        elif initializer == Initializer.CONSTANT:
            init_dist = self.namespace.Settings.InitDistribution.CONSTANT
        elif initializer == Initializer.LONGTAIL:
            init_dist = self.namespace.Settings.InitDistribution.LONGTAIL
        else:
            raise NotImplementedError
        set_field(self.settings, "initDistribution", init_dist)
        set_field(self.settings, "initializer", self.namespace.Settings.InitSet.SIMPLE)

    @property
    def relation_neuron_activation(self) -> Activation:
        return get_field(self.settings, "atomNeuronActivation")

    @relation_neuron_activation.setter
    def relation_neuron_activation(self, value: Activation):
        set_field(self.settings, "atomNeuronActivation", self.get_activation_function(value))

    @property
    def rule_neuron_activation(self) -> Activation:
        return get_field(self.settings, "ruleNeuronActivation")

    @rule_neuron_activation.setter
    def rule_neuron_activation(self, value: Activation):
        set_field(self.settings, "ruleNeuronActivation", self.get_activation_function(value))

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

    def get_activation_function(self, activation: Activation):
        if activation == Activation.SIGMOID:
            return self.namespace.Settings.ActivationFcn.SIGMOID
        if activation == Activation.TANH:
            return self.namespace.Settings.ActivationFcn.TANH
        if activation == Activation.SIGNUM:
            return self.namespace.Settings.ActivationFcn.SIGNUM
        if activation == Activation.RELU:
            return self.namespace.Settings.ActivationFcn.RELU
        if activation == Activation.IDENTITY:
            return self.namespace.Settings.ActivationFcn.IDENTITY
        if activation == Activation.LUKASIEWICZ:
            return self.namespace.Settings.ActivationFcn.LUKASIEWICZ
        if activation == Activation.SOFTMAX:
            return self.namespace.Settings.ActivationFcn.SOFTMAX
        if activation == Activation.SPARSEMAX:
            return self.namespace.Settings.ActivationFcn.SPARSEMAX
        raise NotImplementedError

    def to_json(self) -> str:
        return self.settings.exportToJson()
