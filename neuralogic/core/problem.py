from neuralogic import get_neuralogic, get_gateway
from py4j.java_collections import ListConverter
from py4j.java_gateway import get_field, set_field

from typing import Union, List, Optional, Iterator, Tuple
from contextlib import contextmanager

from neuralogic.core.builder import Builder, Backend
from neuralogic.core.constructs.atom import BaseAtom, WeightedAtom
from neuralogic.core.constructs.rule import Rule
from neuralogic.core.constructs.predicate import PredicateMetadata
from neuralogic.core.constructs.java_objects import get_current_java_factory, set_java_factory, JavaFactory
from neuralogic.core.settings import Settings
from neuralogic.core.sources import Sources
from neuralogic.utils.data import Dataset

TemplateEntries = Union[BaseAtom, WeightedAtom, Rule]


def stream_to_list(stream) -> List:
    return list(stream.collect(get_gateway().jvm.java.util.stream.Collectors.toList()))


class Problem:
    def __init__(self, settings: Optional[Settings] = None):
        if settings is None:
            settings = Settings()

        self.java_factory = JavaFactory(settings)

        self.template: List[TemplateEntries] = []
        self.examples: List[TemplateEntries] = []
        self.queries: List[TemplateEntries] = []

        self.counter = 0

    def add_rule(self, rule):
        self.add_rules([rule])

    def add_rules(self, rules: List):
        self.template.extend(rules)

    def add_example(self, example):
        self.add_examples([example])

    def add_examples(self, examples: List):
        self.examples.extend(examples)

    def add_query(self, query):
        self.add_queries([query])

    def add_queries(self, queries: List):
        self.queries.extend(queries)

    def build_queries(self, query_builder):
        namespace = get_neuralogic().cz.cvut.fel.ida.logic.constructs.example
        logic_samples = []

        for query in self.queries:
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

    def build_examples(self, examples_builder):
        logic_samples = []
        namespace = get_neuralogic().cz.cvut.fel.ida.logic.constructs.example

        for example in self.examples:
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

    def get_parsed_template(self):
        predicate_metadata = []
        weighted_rules = []
        valued_facts = []

        for rule in self.template:
            if isinstance(rule, PredicateMetadata):
                predicate_metadata.append(rule.java_object)
            elif isinstance(rule, Rule):
                weighted_rules.append(rule.java_object)
            elif isinstance(rule, (WeightedAtom, BaseAtom)):
                valued_facts.append(rule.java_object)

        weighted_rules = ListConverter().convert(weighted_rules, get_gateway()._gateway_client)
        valued_facts = ListConverter().convert(valued_facts, get_gateway()._gateway_client)
        predicate_metadata = ListConverter().convert(predicate_metadata, get_gateway()._gateway_client)

        template_namespace = get_neuralogic().cz.cvut.fel.ida.logic.constructs.template.types
        return template_namespace.ParsedTemplate(weighted_rules, valued_facts)

    def merge_queries_with_examples(self, queries, examples):
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

    def build(self, backend: Backend):
        from neuralogic.nn import get_neuralogic_layer

        self.counter = 0

        previous_factory = get_current_java_factory()
        set_java_factory(self.java_factory)

        namespace = get_neuralogic().cz.cvut.fel.ida.logic.constructs.building
        examples_builder = namespace.ExamplesBuilder(self.java_factory.settings.settings)
        query_builder = namespace.QueriesBuilder(self.java_factory.settings.settings)
        query_builder.setFactoriesFrom(examples_builder)

        examples = self.build_examples(examples_builder)
        queries = self.build_queries(query_builder)
        logic_samples = self.merge_queries_with_examples(queries, examples)

        logic_samples = ListConverter().convert(logic_samples, get_gateway()._gateway_client).stream()
        parsed_template = self.get_parsed_template()

        set_java_factory(previous_factory)
        weights, samples = Builder.from_problem(parsed_template, logic_samples, backend, self.java_factory.settings)

        return get_neuralogic_layer(backend)(weights, self.java_factory.settings), Dataset(samples)

    @contextmanager
    def context(self) -> Iterator["Problem"]:
        previous_factory = get_current_java_factory()
        set_java_factory(self.java_factory)
        yield self
        set_java_factory(previous_factory)

    def rules_to_str(self) -> str:
        return "\n".join(str(r) for r in self.template)

    def examples_to_str(self) -> str:
        return "\n".join(str(r) for r in self.examples)

    def queries_to_str(self) -> str:
        return "\n".join(str(r) for r in self.queries)

    @staticmethod
    def build_from_dir(directory: str, backend: Backend, settings: Settings, args: Optional[List] = None):
        from neuralogic.nn import get_neuralogic_layer

        args = [] if args is None else args

        args.extend(["-sd", str(directory)])
        sources = Sources.from_args(args, settings)

        weights, samples = Builder.from_sources(settings, backend, sources)

        return get_neuralogic_layer(backend)(weights, settings), Dataset(samples)

    @staticmethod
    def build_from_files(
        rules_file: str,
        backend: Backend,
        settings: Settings,
        example_file: Optional[str] = None,
        queries_file: Optional[str] = None,
        args: Optional[List] = None,
    ):
        from neuralogic.nn import get_neuralogic_layer

        args = [] if args is None else args

        args.extend(["-t", str(rules_file)])

        if queries_file is not None:
            args.extend(["-q", str(queries_file)])
        if example_file is not None:
            args.extend(["-e", str(example_file)])

        sources = Sources.from_args(args, settings)

        weights, samples = Builder.from_sources(settings, backend, sources)

        return get_neuralogic_layer(backend)(weights, settings), Dataset(samples)
