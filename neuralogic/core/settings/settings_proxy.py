import jpype

import neuralogic
from neuralogic import is_initialized, initialize
from neuralogic.core.constructs.function import Transformation, Combination, Aggregation
from neuralogic.core.enums import Grounder
from neuralogic.nn.init import Initializer
from neuralogic.nn.loss import MSE, SoftEntropy, CrossEntropy, ErrorFunction
from neuralogic.optim import Optimizer


class SettingsProxy:
    _number_format = None

    def __init__(
        self,
        *,
        optimizer: Optimizer,
        learning_rate: float,
        epochs: int,
        error_function: ErrorFunction,
        initializer: Initializer,
        rule_transformation: Transformation,
        rule_combination: Combination,
        rule_aggregation: Aggregation,
        relation_transformation: Transformation,
        relation_combination: Combination,
        iso_value_compression: bool,
        chain_pruning: bool,
        prune_only_identities: bool,
        grounder: Grounder,
    ):
        if not is_initialized():
            initialize()

        self.settings_class = jpype.JClass("cz.cvut.fel.ida.setup.Settings")
        self.settings = self.settings_class()

        self._optimizer = optimizer

        params = locals().copy()
        params.pop("self")

        for key, value in params.items():
            self.__setattr__(key, value)

        self.settings.debugExporting = False
        self.settings.exportBlocks = []

        self.settings.infer()
        self.settings.supressConsoleOutput = True
        self.settings.supressLogFileOutput = True
        self.settings.loggingLevel = jpype.JClass("java.util.logging.Level").OFF

        self._setup_random_generator()

    @staticmethod
    def number_format():
        if SettingsProxy._number_format is None:
            SettingsProxy._number_format = jpype.JClass("cz.cvut.fel.ida.setup.Settings").superDetailedNumberFormat
        return SettingsProxy._number_format

    def _setup_random_generator(self):
        if neuralogic._rnd_generator is None:
            neuralogic._rnd_generator = self.settings.random
            self.settings.random.setSeed(neuralogic._seed)
        self.settings.random = neuralogic._rnd_generator

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
    def prune_only_identities(self) -> bool:
        return self.settings.pruneOnlyIdentities

    @prune_only_identities.setter
    def prune_only_identities(self, prune_only_identities: bool):
        self.settings.pruneOnlyIdentities = prune_only_identities

    @property
    def grounder(self):
        return self.settings.grounding

    @grounder.setter
    def grounder(self, grounder: Grounder):
        if grounder == Grounder.BUP:
            self.settings.grounding = self.settings.GroundingAlgo.BUP
        elif grounder == Grounder.GRINGO:
            self.settings.grounding = self.settings.GroundingAlgo.GRINGO
        else:
            raise ValueError(f"Invalid grounder {grounder}")

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer):
        if optimizer.name() == "SGD":
            java_optimizer = self.settings_class.OptimizerSet.SGD
        elif optimizer.name() == "Adam":
            java_optimizer = self.settings_class.OptimizerSet.ADAM
        else:
            raise NotImplementedError

        self._optimizer = optimizer
        self.settings.setOptimizer(java_optimizer)
        self.settings.initLearningRate = optimizer.lr

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
        self.settings.inferOutputFcns = True

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
                self.settings.inferOutputFcns = False
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
        if not isinstance(initializer, Initializer):
            raise TypeError()

        settings = initializer.get_settings()
        init_name = settings.pop("initializer")

        if initializer.is_simple():
            self.settings.initializer = self.settings_class.InitSet.SIMPLE
            self.settings.initDistribution = getattr(self.settings_class.InitDistribution, init_name)
        else:
            self.settings.initializer = getattr(self.settings_class.InitSet, init_name)

        for key, value in settings.items():
            self.__setattr__(key, value)

    @property
    def relation_transformation(self) -> Transformation:
        return Transformation(str(self.settings.atomNeuronTransformation))

    @relation_transformation.setter
    def relation_transformation(self, value: Transformation):
        self.settings.atomNeuronTransformation = self.get_transformation_function(value)

    @property
    def relation_combination(self) -> Combination:
        return Combination(str(self.settings.atomNeuronCombination))

    @relation_combination.setter
    def relation_combination(self, value: Combination):
        self.settings.atomNeuronCombination = self.get_combination_function(value)

    @property
    def rule_transformation(self) -> Transformation:
        return Transformation(str(self.settings.ruleNeuronTransformation))

    @rule_transformation.setter
    def rule_transformation(self, value: Transformation):
        self.settings.ruleNeuronTransformation = self.get_transformation_function(value)

    @property
    def rule_combination(self) -> Combination:
        return Combination(str(self.settings.ruleNeuronCombination))

    @rule_combination.setter
    def rule_combination(self, value: Combination):
        self.settings.ruleNeuronCombination = self.get_combination_function(value)

    @property
    def rule_aggregation(self) -> Aggregation:
        return Aggregation(str(self.settings.aggNeuronAggregation))

    @rule_aggregation.setter
    def rule_aggregation(self, value: Aggregation):
        self.settings.aggNeuronAggregation = self.get_aggregation_function(value)

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

    def get_combination_function(self, combination: Combination):
        combination_name = str(combination)
        return self.settings_class.parseCombination(combination_name)

    def get_aggregation_function(self, aggregation: Aggregation):
        aggregation_name = str(aggregation)
        return self.settings_class.parseCombination(aggregation_name)

    def get_transformation_function(self, transformation: Transformation):
        transformation_name = str(transformation)
        return self.settings_class.parseTransformation(transformation_name)

    def __setitem__(self, key, value):
        setattr(self.settings, key, value)

    def __getitem__(self, item):
        return getattr(self.settings, item)

    def to_json(self) -> str:
        return self.settings.exportToJson()
