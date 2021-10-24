from neuralogic import get_neuralogic, get_gateway
from py4j.java_collections import ListConverter
from py4j.java_gateway import get_field, set_field

from typing import Union, List, Optional, Set, Dict, Any, Callable

from neuralogic.core.builder import Builder, Backend
from neuralogic.core.constructs.atom import BaseAtom, WeightedAtom
from neuralogic.core.constructs.rule import Rule
from neuralogic.core.constructs.predicate import PredicateMetadata
from neuralogic.core.constructs.java_objects import JavaFactory
from neuralogic.core.settings import Settings
from neuralogic.core.sources import Sources

TemplateEntries = Union[BaseAtom, WeightedAtom, Rule]


class BuiltDataset:
    """BuiltDataset represents an already built dataset - that is, a dataset that has been grounded and neuralized."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]


class Template:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        module_list: Optional = None,
        template_file: Optional[str] = None,
    ):
        if settings is None:
            settings = Settings()

        self.java_factory = None

        self.template: List[TemplateEntries] = []
        self.parsed_template = None

        self.settings = settings
        self.template_file = template_file

        self.locked_template = False
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
        if self.locked_template:
            raise Exception
        if self.parsed_template is not None:
            self.parsed_template = None

        self.template.extend(rules)

    def build_queries(self, queries, query_builder):
        namespace = get_neuralogic().cz.cvut.fel.ida.logic.constructs.example
        logic_samples = []

        for query in queries:
            head, conjunction = self.java_factory.get_query(query)
            facts = get_field(conjunction, "facts")

            if head is not None:
                id = get_field(head, "literal").toString()
                if head.getValue() is None or not isinstance(query.head.weight, (float, int)):
                    logic_samples.extend(
                        [namespace.LogicSample(f.getValue(), query_builder.createQueryAtom(id, f), True) for f in facts]
                    )
                else:
                    importance = get_field(head.getValue(), "value")
                    logic_samples.extend(
                        [
                            namespace.LogicSample(f.getValue(), query_builder.createQueryAtom(id, importance, f), True)
                            for f in facts
                        ]
                    )
            else:
                id = str(self.counter)
                logic_samples.extend(
                    [namespace.LogicSample(f.getValue(), query_builder.createQueryAtom(id, f), True) for f in facts]
                )
            self.counter += 1
        return logic_samples

    def build_examples(self, examples, examples_builder):
        logic_samples = []
        namespace = get_neuralogic().cz.cvut.fel.ida.logic.constructs.example

        for example in examples:
            label, lifted_example = self.java_factory.get_lifted_example(example)

            value = None
            label_fact = None if label is None else get_field(label, "facts")
            label_size = 0 if label is None else label_fact.size()

            if label is None or label_size == 0:
                query_atom = examples_builder.createQueryAtom(str(self.counter), None, lifted_example)
            elif label_size == 1:
                if label_fact.get(0).getValue() is None:
                    literal_string = get_field(label[0], "literal").toString()
                    query_atom = examples_builder.createQueryAtom(literal_string, label_fact.get(0), lifted_example)
                else:
                    value = label_fact.get(0).getValue()
                    query_atom = examples_builder.createQueryAtom(str(self.counter), label_fact.get(0), lifted_example)
            else:
                raise NotImplementedError
            logic_samples.append(namespace.LogicSample(value, query_atom))
            self.counter += 1
        return logic_samples

    def get_parsed_template(self, settings: Settings):
        self.java_factory = JavaFactory()

        if self.template_file is not None:
            return Builder(settings).build_template_from_file(settings, self.template_file)

        predicate_metadata = []
        weighted_rules = []
        valued_facts = []

        for rule in self.template:
            if isinstance(rule, PredicateMetadata):
                predicate_metadata.append(self.java_factory.get_predicate_metadata_pair(rule))
            elif isinstance(rule, Rule):
                weighted_rules.append(self.java_factory.get_rule(rule))
            elif isinstance(rule, (WeightedAtom, BaseAtom)):
                valued_facts.append(self.java_factory.get_valued_fact(rule, self.java_factory.get_variable_factory()))

        weighted_rules = ListConverter().convert(weighted_rules, get_gateway()._gateway_client)
        valued_facts = ListConverter().convert(valued_facts, get_gateway()._gateway_client)
        predicate_metadata = ListConverter().convert(predicate_metadata, get_gateway()._gateway_client)

        template_namespace = get_neuralogic().cz.cvut.fel.ida.logic.constructs.template.types
        return template_namespace.ParsedTemplate(weighted_rules, valued_facts)

    def build(self, backend: Backend, settings: Settings, *, native_backend_models=False):
        from neuralogic.nn import get_neuralogic_layer

        if backend == Backend.PYG:
            return get_neuralogic_layer(backend, native_backend_models)(self.module_list)

        self.parsed_template = self.get_parsed_template(settings)
        model = Builder(settings).build_model(self.parsed_template, backend, settings)

        return get_neuralogic_layer(backend)(model, self.parsed_template, settings)

    def build_dataset(self, dataset, backend: Backend, settings: Settings) -> BuiltDataset:
        """Builds the dataset (does grounding and neuralization) for this template instance and the backend

        :param dataset:
        :param backend:
        :param settings:
        :return:
        """
        if self.parsed_template is None:
            self.parsed_template = self.get_parsed_template(settings)

        weight_factory = self.java_factory.weight_factory
        namespace = get_neuralogic().cz.cvut.fel.ida.logic.constructs.building
        examples_builder = namespace.ExamplesBuilder(settings.settings)
        query_builder = namespace.QueriesBuilder(settings.settings)
        query_builder.setFactoriesFrom(examples_builder)

        if not dataset.file_sources:
            examples = dataset.examples
            queries = dataset.queries

            if dataset.data is not None:
                examples = []
                queries = []

                for data in dataset.data:
                    from neuralogic.utils.templates import TemplateList

                    query, example = TemplateList.to_inputs(self, data.x, data.edge_index, data.y, data.y_mask)
                    examples.append(example)
                    queries.extend(query)

            self.java_factory.weight_factory = self.java_factory.get_new_weight_factory()
            examples = self.build_examples(examples, examples_builder)

            self.java_factory.weight_factory = self.java_factory.get_new_weight_factory()
            queries = self.build_queries(queries, query_builder)

            logic_samples = Template.merge_queries_with_examples(queries, examples)
            logic_samples = ListConverter().convert(logic_samples, get_gateway()._gateway_client).stream()

            samples = Builder(settings).from_logic_samples(self.parsed_template, logic_samples, backend)
        else:
            args = ["-t", dataset.examples_file or dataset.queries_file]
            if dataset.queries_file is not None:
                args.extend(["-q", dataset.queries_file])
            if dataset.examples_file is not None:
                args.extend(["-e", dataset.examples_file])
            sources = Sources.from_args(args, settings)
            samples = Builder(settings).from_sources(self.parsed_template, sources, backend)

        self.java_factory.weight_factory = weight_factory
        return BuiltDataset(samples)

    @staticmethod
    def merge_queries_with_examples(queries, examples):
        logic_samples = []

        if len(examples) == 0:
            return queries

        if len(queries) == 0:
            return examples

        if len(examples) == 1:
            for query in queries:
                example = examples[0] if get_field(get_field(query, "query"), "evidence") is None else query
                query_object = query if get_field(query, "isQueryOnly") else examples[0]

                example_evidence = get_field(get_field(example, "query"), "evidence")
                set_field(get_field(query_object, "query"), "evidence", example_evidence)
                logic_samples.append(query)
            return logic_samples

        if len(examples) != len(queries):
            raise Exception(f"The size of examples {len(examples)} doesn't match the size of queries {len(queries)}")

        for query, example in zip(queries, examples):
            example_object = example if get_field(get_field(query, "query"), "evidence") is None else query
            query_object = query if get_field(query, "isQueryOnly") else example

            example_evidence = get_field(get_field(example_object, "query"), "evidence")
            set_field(get_field(query_object, "query"), "evidence", example_evidence)
            logic_samples.append(query)
        return logic_samples

    def __str__(self) -> str:
        return "\n".join(str(r) for r in self.template)
