from typing import Union, List, Optional, Iterable

import jpype

from neuralogic import is_initialized, initialize
from neuralogic.core.builder import Builder, DatasetBuilder
from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.core.constructs.rule import Rule
from neuralogic.core.constructs.predicate import PredicateMetadata
from neuralogic.core.constructs.java_objects import JavaFactory
from neuralogic.core.neural_module import NeuralModule
from neuralogic.core.settings import SettingsProxy, Settings
from neuralogic.nn.module.module import Module


TemplateEntries = Union[BaseRelation, WeightedRelation, Rule]


class Template(NeuralModule):
    def __init__(self, *, template_file: Optional[str] = None):
        super().__init__()
        self._template: List[TemplateEntries] = []
        self._template_file = template_file

    def add_rule(self, rule) -> None:
        """Adds one rule to the template

        :param rule:
        :return:
        """
        self.add_rules([rule])

    def add_rules(self, rules: List):
        """Adds multiple rules to the template

        :param rules:
        :return:
        """
        self._template.extend(rules)

    def add_module(self, module: Module):
        """Expands the module into rules and adds them into the template

        :param module:
        :return:
        """
        self.add_rules(module())

    def build(self, settings: Settings) -> "Template":
        java_factory = JavaFactory()
        settings_proxy = settings.create_proxy()

        parsed_template = self._get_parsed_template(settings_proxy, java_factory)
        neural_model = Builder(settings_proxy).build_model(parsed_template, settings_proxy)

        self._initialize_neural_module(
            DatasetBuilder(parsed_template, java_factory),
            settings_proxy,
            neural_model,
        )

        return self

    def remove_duplicates(self):
        """Remove duplicates from the template"""
        entries = set()
        deduplicated_template: List[TemplateEntries] = []

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

        if self._template_file is not None:
            return Builder(settings).build_template_from_file(settings, self._template_file)

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

        return template

    def __str__(self) -> str:
        return "\n".join(str(r) for r in self._template)

    def __repr__(self) -> str:
        return self.__str__()

    def __iadd__(self, other) -> "Template":
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
        self._template.pop(key)

    def __setitem__(self, key, value):
        if isinstance(value, (Iterable, Module)):
            raise NotImplementedError
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
