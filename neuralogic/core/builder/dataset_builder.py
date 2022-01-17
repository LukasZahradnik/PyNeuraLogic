from typing import Union, Set, Dict

import jpype

from neuralogic import is_initialized, initialize
from neuralogic.core.builder.builder import Builder
from neuralogic.core.builder.components import BuiltDataset
from neuralogic.core.enums import Backend
from neuralogic.core.constructs.atom import BaseAtom, WeightedAtom
from neuralogic.core.constructs.rule import Rule
from neuralogic.core.constructs.java_objects import JavaFactory
from neuralogic.core.settings import SettingsProxy
from neuralogic.core.sources import Sources
from neuralogic.core.dataset import Dataset

TemplateEntries = Union[BaseAtom, WeightedAtom, Rule]


class DatasetBuilder:
    def __init__(self, parsed_template, java_factory: JavaFactory):
        if not is_initialized():
            initialize()

        self.java_factory = java_factory
        self.parsed_template = parsed_template

        self.grounding_mode = jpype.JClass("cz.cvut.fel.ida.setup.Settings").GroundingMode
        self.logic_sample = jpype.JClass("cz.cvut.fel.ida.logic.constructs.example.LogicSample")

        self.examples_builder = jpype.JClass("cz.cvut.fel.ida.logic.constructs.building.ExamplesBuilder")
        self.queries_builder = jpype.JClass("cz.cvut.fel.ida.logic.constructs.building.QueriesBuilder")

        self.counter = 0
        self.hooks: Dict[str, Set] = {}

    def build_queries(self, queries, query_builder):
        logic_samples = []

        for query in queries:
            head, conjunction = self.java_factory.get_query(query)
            facts = conjunction.facts

            if head is not None:
                id = head.literal.toString()
                if head.getValue() is None or not isinstance(query.head.weight, (float, int)):
                    logic_samples.extend(
                        [self.logic_sample(f.getValue(), query_builder.createQueryAtom(id, f), True) for f in facts]
                    )
                else:
                    importance = head.getValue().value
                    logic_samples.extend(
                        [
                            self.logic_sample(f.getValue(), query_builder.createQueryAtom(id, importance, f), True)
                            for f in facts
                        ]
                    )
            else:
                id = str(self.counter)
                logic_samples.extend(
                    [self.logic_sample(f.getValue(), query_builder.createQueryAtom(id, f), True) for f in facts]
                )
            self.counter += 1
        return logic_samples

    def build_examples(self, examples, examples_builder):
        logic_samples = []
        one = jpype.JClass("cz.cvut.fel.ida.algebra.values.ScalarValue")(1.0)

        for example in examples:
            label, lifted_example = self.java_factory.get_lifted_example(example)

            value = one
            label_fact = None if label is None else label.facts
            label_size = 0 if label is None else label_fact.size()

            if label is None or label_size == 0:
                query_atom = examples_builder.createQueryAtom(str(self.counter), None, lifted_example)
            elif label_size == 1:
                if label_fact.get(0).getValue() is None:
                    literal_string = label_fact.get(0).literal.toString()
                    query_atom = examples_builder.createQueryAtom(literal_string, label_fact.get(0), lifted_example)
                else:
                    value = label_fact.get(0).getValue()
                    query_atom = examples_builder.createQueryAtom(str(self.counter), label_fact.get(0), lifted_example)
            else:
                raise NotImplementedError
            logic_samples.append(self.logic_sample(value, query_atom))
            self.counter += 1
        return logic_samples

    def build_dataset(self, dataset: Dataset, backend: Backend, settings: SettingsProxy) -> BuiltDataset:
        """Builds the dataset (does grounding and neuralization) for this template instance and the backend

        :param dataset:
        :param backend:
        :param settings:
        :return:
        """
        self.counter = 0
        weight_factory = self.java_factory.weight_factory

        examples_builder = self.examples_builder(settings.settings)
        query_builder = self.queries_builder(settings.settings)
        query_builder.setFactoriesFrom(examples_builder)

        settings.settings.groundingMode = self.grounding_mode.INDEPENDENT

        if not dataset.file_sources:
            examples = dataset.examples
            queries = dataset.queries

            if dataset.data is not None:
                examples = []
                queries = []

                for data in dataset.data:
                    query, example = data.to_logic_form(
                        one_hot_encoding=dataset.one_hot_encoding,
                        max_classes=dataset.number_of_classes,
                    )

                    examples.append(example)
                    queries.append(query)

            if len(examples) == 1:
                settings.settings.groundingMode = self.grounding_mode.GLOBAL

            self.java_factory.weight_factory = self.java_factory.get_new_weight_factory()
            examples = self.build_examples(examples, examples_builder)

            self.java_factory.weight_factory = self.java_factory.get_new_weight_factory()
            queries = self.build_queries(queries, query_builder)

            logic_samples = DatasetBuilder.merge_queries_with_examples(queries, examples)
            logic_samples = jpype.java.util.ArrayList(logic_samples).stream()

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
                example = examples[0] if query.query.evidence is None else query
                query_object = query if query.isQueryOnly else examples[0]
                query_object.query.evidence = example.query.evidence
                logic_samples.append(query)
            return logic_samples

        if len(examples) != len(queries):
            raise Exception(f"The size of examples {len(examples)} doesn't match the size of queries {len(queries)}")

        for query, example in zip(queries, examples):
            example_object = example if query.query.evidence is None else query
            query_object = query if query.isQueryOnly else example
            query_object.query.evidence = example_object.query.evidence
            logic_samples.append(query)
        return logic_samples
