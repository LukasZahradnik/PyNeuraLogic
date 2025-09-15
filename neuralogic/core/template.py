from typing import Iterable

import jpype

from neuralogic.dataset import Dataset
from neuralogic.setup import is_initialized, initialize
from neuralogic.core.builder import Builder, DatasetBuilder
from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.core.constructs.rule import Rule
from neuralogic.core.constructs.predicate import PredicateMetadata
from neuralogic.core.constructs.java_objects import JavaFactory
from neuralogic.core.constructs.factories import R
from neuralogic.core.neural_module import NeuralModule
from neuralogic.core.settings import SettingsProxy, Settings
from neuralogic.nn.module.module import Module


TemplateEntries = BaseRelation | WeightedRelation | Rule | PredicateMetadata


class Template(NeuralModule):
    def __init__(self, *, template_file: str | None = None):
        super().__init__()
        self._template: list[TemplateEntries] = []
        self._template_file = template_file

    def add_rule(self, rule) -> None:
        """Adds one rule to the template

        :param rule:
        :return:
        """
        self.add_rules([rule])

    def add_rules(self, rules: list[TemplateEntries]) -> None:
        """Adds multiple rules to the template

        :param rules:
        :return:
        """
        if self._neural_model is not None:
            raise ValueError("Cannot modify built template")
        self._parsed_template = None
        self._template.extend(rules)

    def add_module(self, module: Module):
        """Expands the module into rules and adds them into the template

        :param module:
        :return:
        """
        self.add_rules(module())

    def build(self, settings: Settings | None = None, torch: bool = False) -> "Template":
        java_factory = JavaFactory()
        settings_proxy = settings.create_proxy() if settings is not None else Settings().create_disconnected_proxy()

        parsed_template = self._get_parsed_template(settings_proxy, java_factory)
        neural_model = Builder(settings_proxy).build_model(parsed_template, settings_proxy)

        self._initialize_neural_module(
            DatasetBuilder(parsed_template, java_factory),
            settings_proxy,
            neural_model,
            torch,
        )

        return self

    def remove_duplicates(self):
        """Remove duplicates from the template"""
        if self._neural_model is not None:
            raise ValueError("Cannot modify built template")

        self._parsed_template = None

        entries = set()
        deduplicated_template: list[TemplateEntries] = []

        for entry in self._template:
            entry_str = str(entry)

            if entry_str in entries:
                continue
            entries.add(entry_str)
            deduplicated_template.append(entry)
        self._template = deduplicated_template

    def _get_parsed_template(self, settings: SettingsProxy, java_factory: JavaFactory):
        if not is_initialized():
            initialize()

        if self._parsed_template is not None:
            return self._parsed_template

        if self._template_file is not None:
            self._parsed_template = Builder(settings).build_template_from_file(settings, self._template_file)
            return self._parsed_template

        predicate_metadata = []
        weighted_rules = []
        valued_facts = []

        for rule in self._template:
            if isinstance(rule, PredicateMetadata):
                predicate_metadata.append(java_factory.get_predicate_metadata_pair(rule))
            elif isinstance(rule, Rule):
                weighted_rules.append(java_factory.get_rule(rule))
            elif isinstance(rule, (WeightedRelation, BaseRelation)):
                valued_facts.append(java_factory.get_valued_fact(rule, java_factory.get_variable_factory()))

        parsed_template = jpype.JClass("cz.cvut.fel.ida.logic.constructs.template.types.ParsedTemplate")
        template = parsed_template(jpype.java.util.ArrayList(weighted_rules), jpype.java.util.ArrayList(valued_facts))

        template.weightsMetadata = (jpype.java.util.List) @ jpype.java.util.ArrayList([])
        template.predicatesMetadata = jpype.java.util.ArrayList(predicate_metadata)

        metadata_processor = jpype.JClass("cz.cvut.fel.ida.logic.constructs.template.transforming.MetadataProcessor")
        metadata_processor = metadata_processor(settings.settings)

        metadata_processor.processMetadata(template)
        self._parsed_template = template

        return self._parsed_template

    def derivable_queries(self, example: list[BaseRelation | Rule] | None = None):
        settings = Settings(iso_value_compression=False, chain_pruning=False).create_disconnected_proxy()
        java_factory = JavaFactory()

        parsed_template = self._get_parsed_template(settings, java_factory)
        dataset_builder = DatasetBuilder(parsed_template, java_factory)

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

    def query(self, query: BaseRelation, examples: list[BaseRelation | Rule] | None = None):
        settings = Settings(iso_value_compression=False, chain_pruning=False).create_disconnected_proxy()
        java_factory = JavaFactory()

        parsed_template = self._get_parsed_template(settings, java_factory)
        dataset_builder = DatasetBuilder(parsed_template, java_factory)

        try:
            grounded_dataset = dataset_builder.ground_dataset(Dataset().add(query, examples), settings)
        except Exception:
            return {}

        results = [node.substitutions for sample in grounded_dataset for node in sample.get_atoms(query)]

        if not results:
            return {}
        return results

    def q(self, query: BaseRelation, examples: list[BaseRelation | Rule] | None = None):
        return self.query(query, examples)

    def __str__(self) -> str:
        return "\n".join(str(r) for r in self._template)

    def __repr__(self) -> str:
        return self.__str__()

    def __iadd__(self, other) -> "Template":
        if self._neural_model is not None:
            raise ValueError("Cannot modify built template")
        self._parsed_template = None
        if isinstance(other, Iterable):
            self._template.extend(other)
        elif isinstance(other, Module):
            self._template.extend(other())
        else:
            self._template.append(other)
        return self

    def __getitem__(self, item) -> TemplateEntries:
        return self._template[item]

    def __delitem__(self, key):
        if self._neural_model is not None:
            raise ValueError("Cannot modify built template")
        self._parsed_template = None
        self._template.pop(key)

    def __setitem__(self, key, value):
        if self._neural_model is not None:
            raise ValueError("Cannot modify built template")
        if isinstance(value, (Iterable, Module)):
            raise NotImplementedError
        self._parsed_template = None
        self._template[key] = value

    def __len__(self) -> int:
        return len(self._template)

    def __iter__(self):
        return iter(self._template)

    def __copy__(self) -> "Template":
        temp = Template()

        temp._template_file = self._template_file
        temp._template = [rule for rule in self._template]

        return temp

    def clone(self) -> "Template":
        return self.__copy__()
