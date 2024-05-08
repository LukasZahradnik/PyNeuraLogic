from typing import Union, List, Optional, Set, Dict, Any, Callable, Iterable

import jpype

from neuralogic import is_initialized, initialize
from neuralogic.core.builder import Builder, DatasetBuilder
from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.core.constructs.rule import Rule
from neuralogic.core.constructs.predicate import PredicateMetadata
from neuralogic.core.constructs.java_objects import JavaFactory
from neuralogic.core.settings import SettingsProxy, Settings
from neuralogic.nn.module.module import Module

from neuralogic.utils.visualize import draw_model


TemplateEntries = Union[BaseRelation, WeightedRelation, Rule]


class Template:
    def __init__(
        self,
        *,
        template_file: Optional[str] = None,
    ):
        self.template: List[TemplateEntries] = []
        self.template_file = template_file
        self.hooks: Dict[str, Set] = {}

    def add_hook(self, relation: Union[BaseRelation, str], callback: Callable[[Any], None]) -> None:
        """Hooks the callable to be called with the relation's value as an argument when the value of
        the relation is being calculated.

        :param relation:
        :param callback:
        :return:
        """
        name = str(relation)

        if isinstance(relation, BaseRelation):
            name = name[:-1]

        if name not in self.hooks:
            self.hooks[name] = {callback}
        else:
            self.hooks[name].add(callback)

    def remove_hook(self, relation: Union[BaseRelation, str], callback):
        """Removes the callable from the relation's hooks

        :param relation:
        :param callback:
        :return:
        """
        name = str(relation)

        if isinstance(relation, BaseRelation):
            name = name[:-1]

        if name not in self.hooks:
            return
        self.hooks[name].discard(callback)

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
        self.template.extend(rules)

    def add_module(self, module: Module):
        """Expands the module into rules and adds them into the template

        :param module:
        :return:
        """
        self.add_rules(module())

    def get_parsed_template(self, settings: SettingsProxy, java_factory: JavaFactory):
        if not is_initialized():
            initialize()

        if self.template_file is not None:
            return Builder(settings).build_template_from_file(settings, self.template_file)

        predicate_metadata = []
        weighted_rules = []
        valued_facts = []

        for rule in self.template:
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

    def remove_duplicates(self):
        """Remove duplicates from the template"""
        entries = set()
        deduplicated_template: List[TemplateEntries] = []

        for entry in self.template:
            entry_str = str(entry)

            if entry_str in entries:
                continue
            entries.add(entry_str)
            deduplicated_template.append(entry)
        self.template = deduplicated_template

    def build(self, settings: Settings):
        from neuralogic.nn import get_neuralogic_layer

        java_factory = JavaFactory()
        settings_proxy = settings.create_proxy()

        parsed_template = self.get_parsed_template(settings_proxy, java_factory)
        model = Builder(settings_proxy).build_model(parsed_template, settings_proxy)

        return get_neuralogic_layer()(model, DatasetBuilder(parsed_template, java_factory), self, settings_proxy)

    def draw(
        self,
        filename: Optional[str] = None,
        show=True,
        img_type="png",
        value_detail: int = 0,
        graphviz_path: Optional[str] = None,
        model=None,
        *args,
        **kwargs,
    ):
        if model is None:
            model = self.build(Settings())
        return draw_model(model, filename, show, img_type, value_detail, graphviz_path, *args, **kwargs)

    def __str__(self) -> str:
        return "\n".join(str(r) for r in self.template)

    def __repr__(self) -> str:
        return self.__str__()

    def __iadd__(self, other) -> "Template":
        if isinstance(other, Iterable):
            self.template.extend(other)
        elif isinstance(other, Module):
            self.template.extend(other())
        else:
            self.template.append(other)
        return self

    def __getitem__(self, item) -> TemplateEntries:
        return self.template[item]

    def __delitem__(self, key):
        self.template.pop(key)

    def __setitem__(self, key, value):
        if isinstance(value, (Iterable, Module)):
            raise NotImplementedError
        self.template[key] = value

    def __copy__(self) -> "Template":
        temp = Template()

        temp.template_file = self.template_file
        temp.template = self.template

        return temp

    def clone(self) -> "Template":
        return self.__copy__()
