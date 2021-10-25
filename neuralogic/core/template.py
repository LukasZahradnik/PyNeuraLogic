from neuralogic import get_neuralogic, get_gateway
from py4j.java_collections import ListConverter

from typing import Union, List, Optional, Set, Dict, Any, Callable

from neuralogic.core.builder import Builder, DatasetBuilder
from neuralogic.core.enums import Backend
from neuralogic.core.constructs.atom import BaseAtom, WeightedAtom
from neuralogic.core.constructs.rule import Rule
from neuralogic.core.constructs.predicate import PredicateMetadata
from neuralogic.core.constructs.java_objects import JavaFactory
from neuralogic.core.settings import SettingsProxy, Settings

TemplateEntries = Union[BaseAtom, WeightedAtom, Rule]


class Template:
    def __init__(
        self,
        *,
        module_list: Optional = None,
        template_file: Optional[str] = None,
    ):
        self.template: List[TemplateEntries] = []
        self.template_file = template_file

        self.counter = 0
        self.module_list = None

        if module_list is not None and template_file is None:
            self.module_list = module_list
            module_list.build(self)
            self.locked_template = True

        self.hooks: Dict[str, Set] = {}

    def add_hook(self, atom: Union[BaseAtom, str], callback: Callable[[Any], None]) -> None:
        """Hooks the callable to be called with the atom's value as an argument when the value of
        the atom is being calculated.

        :param atom:
        :param callback:
        :return:
        """
        name = str(atom)

        if isinstance(atom, BaseAtom):
            name = name[:-1]

        if name not in self.hooks:
            self.hooks[name] = {callback}
        else:
            self.hooks[name].add(callback)

    def remove_hook(self, atom: Union[BaseAtom, str], callback):
        """Removes the callable from the atom's hooks

        :param atom:
        :param callback:
        :return:
        """
        name = str(atom)

        if isinstance(atom, BaseAtom):
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

    def get_parsed_template(self, settings: SettingsProxy, java_factory: JavaFactory):
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
            elif isinstance(rule, (WeightedAtom, BaseAtom)):
                valued_facts.append(java_factory.get_valued_fact(rule, java_factory.get_variable_factory()))

        weighted_rules = ListConverter().convert(weighted_rules, get_gateway()._gateway_client)
        valued_facts = ListConverter().convert(valued_facts, get_gateway()._gateway_client)
        predicate_metadata = ListConverter().convert(predicate_metadata, get_gateway()._gateway_client)

        template_namespace = get_neuralogic().cz.cvut.fel.ida.logic.constructs.template.types
        return template_namespace.ParsedTemplate(weighted_rules, valued_facts)

    def build(self, backend: Backend, settings: Settings, *, native_backend_models=False):
        from neuralogic.nn import get_neuralogic_layer

        if backend == Backend.PYG:
            return get_neuralogic_layer(backend, native_backend_models)(self.module_list)

        java_factory = JavaFactory()
        settings_proxy = settings.create_proxy()

        parsed_template = self.get_parsed_template(settings_proxy, java_factory)
        model = Builder(settings_proxy).build_model(parsed_template, backend, settings_proxy)

        return get_neuralogic_layer(backend)(model, DatasetBuilder(parsed_template, java_factory), settings_proxy)

    def __str__(self) -> str:
        return "\n".join(str(r) for r in self.template)
