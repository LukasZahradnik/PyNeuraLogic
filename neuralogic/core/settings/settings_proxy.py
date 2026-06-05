from typing import Any

import jpype

import neuralogic
from neuralogic.core.constructs.function import Aggregation, Combination, Transformation
from neuralogic.core.constructs.function.function import (
    AggregationFunction,
    CombinationFunction,
    TransformationFunction,
)
from neuralogic.core.enums import Grounder
from neuralogic.nn.init import Initializer
from neuralogic.nn.loss import MSE, CrossEntropy, ErrorFunction, SoftEntropy
from neuralogic.nn.optim import Optimizer
from neuralogic.setup import initialize, is_initialized


class SettingsProxy:
    """
    Proxy class for the Java Settings object.

    It provides a Pythonic interface to configure various parameters of the NeuraLogic backend,
    such as optimizers, initializers, error functions, and grounding algorithms.
    """

    def __init__(
        self,
        *,
        optimizer: Optimizer,
        epochs: int,
        error_function: ErrorFunction,
        initializer: Initializer,
        iso_value_compression: bool,
        chain_pruning: bool,
        prune_only_identities: bool,
        grounder: Grounder,
    ):
        """
        Parameters
        ----------
        optimizer : Optimizer
            The optimizer to use for training.
        epochs : int
            The number of training epochs.
        error_function : ErrorFunction
            The error function to use.
        initializer : Initializer
            The weight initializer.
        iso_value_compression : bool
            Whether to use iso-value compression.
        chain_pruning : bool
            Whether to use chain pruning.
        prune_only_identities : bool
            Whether to prune only identity functions.
        grounder : Grounder
            The grounding algorithm to use.
        """
        if not is_initialized():
            initialize()

        self.settings_class = jpype.JClass("cz.cvut.fel.ida.setup.Settings")
        self.settings = self.settings_class()

        self._optimizer = optimizer

        params = locals().copy()
        params.pop("self")

        for key, value in params.items():
            self.__setattr__(key, value)

        self.rule_transformation = Transformation.IDENTITY
        self.relation_transformation = Transformation.IDENTITY
        self.rule_combination = Combination.SUM
        self.relation_combination = Combination.SUM
        self.rule_aggregation = Aggregation.AVG

        self.settings.debugExporting = False
        self.settings.exportBlocks = []

        self.settings.infer()
        self.settings.supressConsoleOutput = True
        self.settings.supressLogFileOutput = True
        self.settings.loggingLevel = jpype.JClass("java.util.logging.Level").OFF

        self.settings.possibleNeuronSharing = True
        self.settings.logGC = False

        self._setup_random_generator()

    def _setup_random_generator(self) -> None:
        if neuralogic.setup._rnd_generator is None:
            neuralogic.setup._rnd_generator = self.settings.random
            self.settings.random.setSeed(neuralogic.setup._seed)
        self.settings.random = neuralogic.setup._rnd_generator

    @property
    def iso_value_compression(self) -> bool:
        """Whether to use iso-value compression."""
        return self.settings.isoValueCompression

    @iso_value_compression.setter
    def iso_value_compression(self, iso_value_compression: bool) -> None:
        self.settings.isoValueCompression = iso_value_compression

    @property
    def chain_pruning(self) -> bool:
        """Whether to use chain pruning (reducing redundant chains of operations)."""
        return self.settings.chainPruning

    @chain_pruning.setter
    def chain_pruning(self, chain_pruning: bool) -> None:
        self.settings.chainPruning = chain_pruning

    @property
    def prune_only_identities(self) -> bool:
        return self.settings.pruneOnlyIdentities

    @prune_only_identities.setter
    def prune_only_identities(self, prune_only_identities: bool) -> None:
        self.settings.pruneOnlyIdentities = prune_only_identities

    @property
    def grounder(self) -> Any:
        """The grounding algorithm to use."""
        return self.settings.grounding

    @grounder.setter
    def grounder(self, grounder: Grounder) -> None:
        if grounder == Grounder.BUP:
            self.settings.grounding = self.settings.GroundingAlgo.BUP
        elif grounder == Grounder.GRINGO:
            self.settings.grounding = self.settings.GroundingAlgo.GRINGO
        else:
            raise ValueError(f"Invalid grounder {grounder}")

    @property
    def optimizer(self) -> Optimizer:
        """The optimizer used for training."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
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
    def initializer_const(self) -> float:
        return self.settings.constantInitValue

    @initializer_const.setter
    def initializer_const(self, value: float) -> None:
        self.settings.constantInitValue = value

    @property
    def initializer_uniform_scale(self) -> float:
        return self.settings.randomInitScale

    @initializer_uniform_scale.setter
    def initializer_uniform_scale(self, value: float) -> None:
        self.settings.randomInitScale = value

    @property
    def error_function(self) -> Any:
        """The error function used for training."""
        return self.settings.errorFunction

    @error_function.setter
    def error_function(self, error_function: ErrorFunction) -> None:
        self.settings.inferOutputFcns = False

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
                self.settings.inferOutputFcns = True
                self.settings.squishLastLayer = False
                java_error_function = self.settings_class.ErrorFcn.CROSSENTROPY
        else:
            raise NotImplementedError

        self.settings.errorFunction = java_error_function

    @property
    def epochs(self) -> int:
        """The maximum number of training epochs."""
        return self.settings.maxCumEpochCount

    @epochs.setter
    def epochs(self, epochs: int) -> None:
        self.settings.maxCumEpochCount = epochs

    @property
    def initializer(self) -> Any:
        """The weight initializer used for model parameters."""
        initializer = self.settings.initializer

        if str(initializer) != "SIMPLE":
            return initializer
        return self.settings.initDistribution

    @initializer.setter
    def initializer(self, initializer: Initializer) -> None:
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
    def relation_transformation(self) -> TransformationFunction:
        return TransformationFunction(str(self.settings.atomNeuronTransformation))

    @relation_transformation.setter
    def relation_transformation(self, value: Transformation) -> None:
        self.settings.atomNeuronTransformation = self.get_transformation_function(value)

    @property
    def relation_combination(self) -> CombinationFunction:
        return CombinationFunction(str(self.settings.atomNeuronCombination))

    @relation_combination.setter
    def relation_combination(self, value: Combination) -> None:
        self.settings.atomNeuronCombination = self.get_combination_function(value)

    @property
    def rule_transformation(self) -> TransformationFunction:
        return TransformationFunction(str(self.settings.ruleNeuronTransformation))

    @rule_transformation.setter
    def rule_transformation(self, value: Transformation) -> None:
        self.settings.ruleNeuronTransformation = self.get_transformation_function(value)

    @property
    def rule_combination(self) -> CombinationFunction:
        return CombinationFunction(str(self.settings.ruleNeuronCombination))

    @rule_combination.setter
    def rule_combination(self, value: Combination) -> None:
        self.settings.ruleNeuronCombination = self.get_combination_function(value)

    @property
    def rule_aggregation(self) -> AggregationFunction:
        return AggregationFunction(str(self.settings.aggNeuronAggregation))

    @rule_aggregation.setter
    def rule_aggregation(self, value: Aggregation) -> None:
        self.settings.aggNeuronAggregation = self.get_aggregation_function(value)

    @property
    def debug_exporting(self) -> bool:
        return self.settings.debugExporting

    @debug_exporting.setter
    def debug_exporting(self, debug_export: bool) -> None:
        self.settings.debugExporting = debug_export

    @property
    def default_fact_value(self) -> float:
        return self.settings.defaultFactValue

    @default_fact_value.setter
    def default_fact_value(self, value: float) -> None:
        self.settings.defaultFactValue = value

    def get_combination_function(self, combination: Combination) -> Any:
        """Returns the Java combination function for the given Python enum value.

        Parameters
        ----------
        combination : Combination
            The combination function enum value.

        Returns
        -------
        Any
            The Java combination function object.
        """
        combination_name = str(combination)
        return self.settings_class.parseCombination(combination_name)

    def get_aggregation_function(self, aggregation: Aggregation) -> Any:
        aggregation_name = str(aggregation)
        return self.settings_class.parseCombination(aggregation_name)

    def get_transformation_function(self, transformation: Transformation) -> Any:
        transformation_name = str(transformation)
        return self.settings_class.parseTransformation(transformation_name)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self.settings, key, value)

    def __getitem__(self, item: str) -> Any:
        return getattr(self.settings, item)

    def to_json(self) -> str:
        """Exports the settings to a JSON string.

        Returns
        -------
        str
            The JSON representation of the settings.
        """
        return self.settings.exportToJson()
