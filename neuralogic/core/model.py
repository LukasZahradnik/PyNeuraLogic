import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jpype

from neuralogic.core.builder import Builder, DatasetBuilder
from neuralogic.core.constructs.factories import R
from neuralogic.core.constructs.java_objects import JavaFactory
from neuralogic.core.constructs.predicate import PredicateMetadata
from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.core.constructs.rule import Rule
from neuralogic.core.neural_module import NeuralModule
from neuralogic.core.settings import Settings, SettingsProxy
from neuralogic.dataset import Dataset
from neuralogic.exceptions import ModelError
from neuralogic.nn.module.module import Module
from neuralogic.setup import initialize, is_initialized

ModelEntries = BaseRelation | WeightedRelation | Rule | PredicateMetadata


class Model(NeuralModule):
    """
    Model is a collection of rules and relations that define the structure of the neural model.
    """

    def __init__(self, *, model_file: str | None = None):
        """
        Parameters
        ----------
        model_file : str, optional
            Path to a model file to load. Default: None.
        """
        super().__init__()
        self._model: list[ModelEntries] = []
        self._model_file = model_file

    def add_rule(self, rule: ModelEntries) -> None:
        """Adds one rule to the model.

        Parameters
        ----------
        rule : ModelEntries
            The rule to add.
        """
        self.add_rules([rule])

    def add_rules(self, rules: list[ModelEntries]) -> None:
        """Adds multiple rules to the model.

        Parameters
        ----------
        rules : list[ModelEntries]
            The rules to add.
        """
        if self._neural_model is not None:
            raise ModelError("Cannot modify built model")
        self._parsed_model = None
        self._model.extend(rules)

    def add_module(self, module: Module) -> None:
        """Expands the module into rules and adds them into the model.

        Parameters
        ----------
        module : Module
            The module to expand and add.
        """
        self.add_rules(module())

    def build(self, settings: Settings | None = None, torch: bool = False) -> "Model":
        """Builds the model into a neural model.

        Parameters
        ----------
        settings : Settings, optional
            The settings for building. Default: None.
        torch : bool
            Whether to use PyTorch backend. Default: False.

        Returns
        -------
        Model
            The built model (self).
        """
        java_factory = JavaFactory()
        settings_proxy = settings.create_proxy() if settings is not None else Settings().create_disconnected_proxy()

        parsed_model = self._get_parsed_model(settings_proxy, java_factory)
        neural_model = Builder(settings_proxy).build_model(parsed_model, settings_proxy)

        self._initialize_neural_module(
            DatasetBuilder(parsed_model, java_factory),
            settings_proxy,
            neural_model,
            torch,
        )

        return self

    def remove_duplicates(self) -> None:
        """Removes duplicates from the model."""
        if self._neural_model is not None:
            raise ModelError("Cannot modify built model")

        self._parsed_model = None

        entries = set()
        deduplicated_model: list[ModelEntries] = []

        for entry in self._model:
            entry_str = str(entry)

            if entry_str in entries:
                continue
            entries.add(entry_str)
            deduplicated_model.append(entry)
        self._model = deduplicated_model

    def _get_parsed_model(self, settings: SettingsProxy, java_factory: JavaFactory) -> Any:
        if not is_initialized():
            initialize()

        if self._parsed_model is not None:
            return self._parsed_model

        if self._model_file is not None:
            self._parsed_model = Builder(settings).build_model_from_file(settings, self._model_file)
            return self._parsed_model

        predicate_metadata = []
        weighted_rules = []
        valued_facts = []

        for rule in self._model:
            if isinstance(rule, PredicateMetadata):
                predicate_metadata.append(java_factory.get_predicate_metadata_pair(rule))
            elif isinstance(rule, Rule):
                weighted_rules.append(java_factory.get_rule(rule))
            elif isinstance(rule, (WeightedRelation, BaseRelation)):
                valued_facts.append(java_factory.get_valued_fact(rule, java_factory.get_variable_factory()))

        parsed_template = jpype.JClass("cz.cvut.fel.ida.logic.constructs.template.types.ParsedTemplate")
        model = parsed_template(jpype.java.util.ArrayList(weighted_rules), jpype.java.util.ArrayList(valued_facts))

        model.weightsMetadata = (jpype.java.util.List) @ jpype.java.util.ArrayList([])
        model.predicatesMetadata = jpype.java.util.ArrayList(predicate_metadata)

        metadata_processor = jpype.JClass("cz.cvut.fel.ida.logic.constructs.template.transforming.MetadataProcessor")
        metadata_processor = metadata_processor(settings.settings)

        metadata_processor.processMetadata(model)
        self._parsed_model = model

        return self._parsed_model

    def derivable_queries(self, example: list[BaseRelation | Rule] | None = None) -> list[BaseRelation] | dict:
        """Returns all derivable queries for the provided example.

        Parameters
        ----------
        example : list[BaseRelation | Rule], optional
            The example to derive queries from. Default: None.

        Returns
        -------
        list[BaseRelation] | dict
            The list of derivable queries.
        """
        settings = Settings(iso_value_compression=False, chain_pruning=False).create_disconnected_proxy()
        java_factory = JavaFactory()

        parsed_model = self._get_parsed_model(settings, java_factory)
        dataset_builder = DatasetBuilder(parsed_model, java_factory)

        try:
            grounded_dataset = dataset_builder.ground_dataset(Dataset().add(None, example), settings)
        except Exception:
            return {}

        results = [
            R.get(name)(sub)
            for sample in grounded_dataset
            for name, substitution in sample.atoms.items()
            for sub in substitution.keys()
        ]

        if not results:
            return {}
        return results

    def query(self, query: BaseRelation, examples: list[BaseRelation | Rule] | None = None) -> list[dict] | dict:
        """Performs a query on the model with the provided examples.

        Parameters
        ----------
        query : BaseRelation
            The query to perform.
        examples : list[BaseRelation | Rule], optional
            The examples to use for the query. Default: None.

        Returns
        -------
        list[dict] | dict
            The list of query results (substitutions).
        """
        settings = Settings(iso_value_compression=False, chain_pruning=False).create_disconnected_proxy()
        java_factory = JavaFactory()

        parsed_model = self._get_parsed_model(settings, java_factory)
        dataset_builder = DatasetBuilder(parsed_model, java_factory)

        try:
            grounded_dataset = dataset_builder.ground_dataset(Dataset().add(query, examples), settings)
        except Exception:
            return {}

        results = [node.substitutions for sample in grounded_dataset for node in sample.get_atoms(query)]

        if not results:
            return {}
        return results

    def q(self, query: BaseRelation, examples: list[BaseRelation | Rule] | None = None) -> list[dict] | dict:
        return self.query(query, examples)

    def __str__(self) -> str:
        return "\n".join(str(r) for r in self._model)

    def __repr__(self) -> str:
        return self.__str__()

    def __iadd__(self, other: ModelEntries | Iterable[ModelEntries] | Module) -> "Model":
        if self._neural_model is not None:
            raise ModelError("Cannot modify built model")
        self._parsed_model = None
        if isinstance(other, Iterable):
            self._model.extend(other)
        elif isinstance(other, Module):
            self._model.extend(other())
        else:
            self._model.append(other)
        return self

    def __getitem__(self, item: int) -> ModelEntries:
        return self._model[item]

    def __delitem__(self, key: int) -> None:
        if self._neural_model is not None:
            raise ModelError("Cannot modify built model")
        self._parsed_model = None
        self._model.pop(key)

    def __setitem__(self, key: int, value: ModelEntries) -> None:
        if self._neural_model is not None:
            raise ModelError("Cannot modify built model")
        if isinstance(value, (Iterable, Module)):
            raise TypeError(f"Cannot set model item to {type(value).__name__}; use add_rules() for multiple entries")
        self._parsed_model = None
        self._model[key] = value

    def __len__(self) -> int:
        return len(self._model)

    def __iter__(self) -> Iterable[ModelEntries]:
        return iter(self._model)

    def __copy__(self) -> "Model":
        mod = Model()

        mod._model_file = self._model_file
        mod._model = [rule for rule in self._model]

        return mod

    def clone(self) -> "Model":
        return self.__copy__()

    def save(self, path: str | Path) -> None:
        """Save the model weights to a file.

        The model must be built before calling :meth:`save`.
        To restore, rebuild the same architecture and call :meth:`load`.

        Parameters
        ----------
        path : str | Path
            Path to the output file (``.pkl`` extension recommended).
        """
        if self._neural_model is None:
            raise ModelError("Cannot save an unbuilt model. Call model.build() first.")
        path = Path(path)
        data = {"state_dict": self.state_dict()}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str | Path) -> None:
        """Load model weights from a file saved by :meth:`save`.

        The model must already be built with the **same architecture**
        that produced the saved file.

        Parameters
        ----------
        path : str | Path
            Path to the saved file.
        """
        if self._neural_model is None:
            raise ModelError("Cannot load weights into an unbuilt model. Call model.build() first.")
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.load_state_dict(data["state_dict"])
