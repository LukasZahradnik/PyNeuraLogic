import jpype

from neuralogic import is_initialized, initialize
from neuralogic.core.enums import Optimizer, Initializer, Activation
from neuralogic.core.error_function import MSE, SoftEntropy, CrossEntropy, ErrorFunction


class SettingsProxy:
    def __init__(
        self,
        *,
        optimizer: Optimizer,
        learning_rate: float,
        epochs: int,
        error_function: ErrorFunction,
        initializer: Initializer,
        initializer_const: float,
        initializer_uniform_scale: float,
        rule_neuron_activation: Activation,
        relation_neuron_activation: Activation,
        iso_value_compression: bool,
        chain_pruning: bool,
    ):
        if not is_initialized():
            initialize()

        self.settings_class = jpype.JClass("cz.cvut.fel.ida.setup.Settings")
        self.settings = self.settings_class()

        params = locals().copy()
        params.pop("self")

        for key, value in params.items():
            self.__setattr__(key, value)

        self.settings.debugExporting = False
        self.settings.exportBlocks = []

        self.settings.infer()

    @property
    def iso_value_compression(self) -> bool:
        return self.settings.isoValueCompression

    @iso_value_compression.setter
    def iso_value_compression(self, iso_value_compression: bool):
        self.settings.isoValueCompression = iso_value_compression

    @property
    def chain_pruning(self) -> bool:
        return self.settings.chainPruning

    @chain_pruning.setter
    def chain_pruning(self, chain_pruning: bool):
        self.settings.chainPruning = chain_pruning

    @property
    def seed(self) -> int:
        return self.settings.seed

    @seed.setter
    def seed(self, seed: int):
        self.settings.seed = seed

    @property
    def learning_rate(self) -> float:
        return self.settings.initLearningRate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        self.settings.initLearningRate = learning_rate

    @property
    def optimizer(self):
        return self.settings.getOptimizer()

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer):
        if optimizer == Optimizer.SGD:
            java_optimizer = self.settings_class.OptimizerSet.SGD
        elif optimizer == Optimizer.ADAM:
            java_optimizer = self.settings_class.OptimizerSet.ADAM
        else:
            raise NotImplementedError
        self.settings.setOptimizer(java_optimizer)

    @property
    def initializer_const(self):
        return self.settings.constantInitValue

    @initializer_const.setter
    def initializer_const(self, value: float):
        self.settings.constantInitValue = value

    @property
    def initializer_uniform_scale(self):
        return self.settings.randomInitScale

    @initializer_uniform_scale.setter
    def initializer_uniform_scale(self, value: float):
        self.settings.randomInitScale = value

    @property
    def error_function(self):
        return self.settings.errorFunction

    @error_function.setter
    def error_function(self, error_function: ErrorFunction):
        if isinstance(error_function, MSE):
            self.settings.squishLastLayer = False
            self.settings.trainOnlineResultsType = self.settings_class.ResultsType.REGRESSION
            java_error_function = self.settings_class.ErrorFcn.SQUARED_DIFF
        elif isinstance(error_function, SoftEntropy):
            self.settings.squishLastLayer = True
            self.settings.trainOnlineResultsType = self.settings_class.ResultsType.CLASSIFICATION
            java_error_function = self.settings_class.ErrorFcn.SOFTENTROPY
        elif isinstance(error_function, CrossEntropy):
            self.settings.trainOnlineResultsType = self.settings_class.ResultsType.CLASSIFICATION

            if error_function.with_logits:
                self.settings.squishLastLayer = True
                java_error_function = self.settings_class.ErrorFcn.SOFTENTROPY
            else:
                self.settings.squishLastLayer = False
                java_error_function = self.settings_class.ErrorFcn.CROSSENTROPY
        else:
            raise NotImplementedError

        self.settings.errorFunction = java_error_function

    @property
    def epochs(self) -> int:
        return self.settings.maxCumEpochCount

    @epochs.setter
    def epochs(self, epochs: int):
        self.settings.maxCumEpochCount = epochs

    @property
    def initializer(self):
        initializer = self.settings.initializer

        if str(initializer) != "SIMPLE":
            return initializer
        return self.settings.initDistribution

    @initializer.setter
    def initializer(self, initializer: Initializer):
        if initializer == Initializer.HE:
            self.settings.initializer = self.settings_class.InitSet.HE
            return
        if initializer == Initializer.GLOROT:
            self.settings.initializer = self.settings_class.InitSet.GLOROT
            return

        if initializer == Initializer.NORMAL:
            init_dist = self.settings_class.InitDistribution.NORMAL
        elif initializer == Initializer.UNIFORM:
            init_dist = self.settings_class.InitDistribution.UNIFORM
        elif initializer == Initializer.CONSTANT:
            init_dist = self.settings_class.InitDistribution.CONSTANT
        elif initializer == Initializer.LONGTAIL:
            init_dist = self.settings_class.InitDistribution.LONGTAIL
        else:
            raise NotImplementedError
        self.settings.initDistribution = init_dist
        self.settings.initializer = self.settings_class.InitSet.SIMPLE

    @property
    def relation_neuron_activation(self) -> Activation:
        return self.settings.atomNeuronActivation

    @relation_neuron_activation.setter
    def relation_neuron_activation(self, value: Activation):
        self.settings.atomNeuronActivation = self.get_activation_function(value)

    @property
    def rule_neuron_activation(self) -> Activation:
        return self.settings.ruleNeuronActivation

    @rule_neuron_activation.setter
    def rule_neuron_activation(self, value: Activation):
        self.settings.ruleNeuronActivation = self.get_activation_function(value)

    @property
    def debug_exporting(self) -> bool:
        return self.settings.debugExporting

    @debug_exporting.setter
    def debug_exporting(self, debug_export: bool):
        self.settings.debugExporting = debug_export

    @property
    def default_fact_value(self) -> float:
        return self.settings.defaultFactValue

    @default_fact_value.setter
    def default_fact_value(self, value: float):
        self.settings.defaultFactValue = value

    def get_activation_function(self, activation: Activation):
        if activation == Activation.SIGMOID:
            return self.settings_class.ActivationFcn.SIGMOID
        if activation == Activation.TANH:
            return self.settings_class.ActivationFcn.TANH
        if activation == Activation.SIGNUM:
            return self.settings_class.ActivationFcn.SIGNUM
        if activation == Activation.RELU:
            return self.settings_class.ActivationFcn.RELU
        if activation == Activation.IDENTITY:
            return self.settings_class.ActivationFcn.IDENTITY
        if activation == Activation.LUKASIEWICZ:
            return self.settings_class.ActivationFcn.LUKASIEWICZ
        if activation == Activation.SOFTMAX:
            return self.settings_class.ActivationFcn.SOFTMAX
        if activation == Activation.SPARSEMAX:
            return self.settings_class.ActivationFcn.SPARSEMAX
        raise NotImplementedError

    def to_json(self) -> str:
        return self.settings.exportToJson()
