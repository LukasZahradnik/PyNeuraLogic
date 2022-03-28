from typing import Union, List, Optional, Set, Dict, Any, Callable, Iterable

import jpype

from neuralogic import is_initialized, initialize
from neuralogic.core.builder import Builder, DatasetBuilder
from neuralogic.core.enums import Backend
from neuralogic.core.constructs.atom import BaseAtom, WeightedAtom
from neuralogic.core.constructs.rule import Rule
from neuralogic.core.constructs.predicate import PredicateMetadata
from neuralogic.core.constructs.java_objects import JavaFactory
from neuralogic.core.settings import SettingsProxy, Settings
from neuralogic.nn.module.module import Module

from neuralogic.utils.visualize import draw_model


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
            elif isinstance(rule, (WeightedAtom, BaseAtom)):
                valued_facts.append(java_factory.get_valued_fact(rule, java_factory.get_variable_factory()))

        parsed_template = jpype.JClass("cz.cvut.fel.ida.logic.constructs.template.types.ParsedTemplate")
        template = parsed_template(jpype.java.util.ArrayList(weighted_rules), jpype.java.util.ArrayList(valued_facts))

        template.weightsMetadata = (jpype.java.util.List) @ jpype.java.util.ArrayList([])
        template.predicatesMetadata = jpype.java.util.ArrayList(predicate_metadata)

        metadata_processor = jpype.JClass("cz.cvut.fel.ida.logic.constructs.template.transforming.MetadataProcessor")
        metadata_processor = metadata_processor(settings.settings)

        metadata_processor.processMetadata(template)
        template.inferTemplateFacts()

        return template

    def build(self, backend: Backend, settings: Settings):
        from neuralogic.nn import get_neuralogic_layer

        if backend == Backend.PYG:
            return get_neuralogic_layer(backend)(self.module_list)

        java_factory = JavaFactory()
        settings_proxy = settings.create_proxy()

        parsed_template = self.get_parsed_template(settings_proxy, java_factory)
        model = Builder(settings_proxy).build_model(parsed_template, backend, settings_proxy)

        return get_neuralogic_layer(backend)(model, DatasetBuilder(parsed_template, java_factory), settings_proxy)

    def draw(
        self,
        filename: Optional[str] = None,
        draw_ipython=True,
        img_type="png",
        value_detail: int = 0,
        graphviz_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        from neuralogic.nn import get_neuralogic_layer

        settings_proxy = Settings().create_proxy()
        java_factory = JavaFactory()

        parsed_template = self.get_parsed_template(settings_proxy, java_factory)
        model = Builder(settings_proxy).build_model(parsed_template, Backend.JAVA, settings_proxy)
        layer = get_neuralogic_layer(Backend.JAVA)(model, DatasetBuilder(parsed_template, java_factory), settings_proxy)

        return draw_model(layer, filename, draw_ipython, img_type, value_detail, graphviz_path, *args, **kwargs)

    def __str__(self) -> str:
        return "\n".join(str(r) for r in self.template)

    def __iadd__(self, other) -> "Template":
        if isinstance(other, Iterable):
            self.template.extend(other)
        elif isinstance(other, Module):
            self.template.extend(other())
        else:
            self.template.append(other)
        return self
