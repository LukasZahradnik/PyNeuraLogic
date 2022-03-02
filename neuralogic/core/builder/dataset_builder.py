import tempfile
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

        self.query_counter = 0
        self.examples_counter = 0

        self.hooks: Dict[str, Set] = {}

    def build_queries(self, queries, query_builder):
        logic_samples = []
        one_query_per_example = True

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
                id = str(self.query_counter)
                if len(facts) > 1:
                    one_query_per_example = False

                logic_samples.extend(
                    [self.logic_sample(f.getValue(), query_builder.createQueryAtom(id, f), True) for f in facts]
                )
            self.query_counter += 1
        return logic_samples, one_query_per_example

    def build_examples(self, examples, examples_builder):
        logic_samples = []
        one = jpype.JClass("cz.cvut.fel.ida.algebra.values.ScalarValue")(1.0)
        examples_queries = False

        for example in examples:
            label, lifted_example = self.java_factory.get_lifted_example(example)
            example_query = False

            value = one
            label_fact = None if label is None else label.facts
            label_size = 0 if label is None else label_fact.size()

            if label is None or label_size == 0:
                query_atom = examples_builder.createQueryAtom(str(self.examples_counter), None, lifted_example)
            elif label_size == 1:
                example_query = True
                if label_fact.get(0).getValue() is None:
                    literal_string = label_fact.get(0).literal.toString()
                    query_atom = examples_builder.createQueryAtom(literal_string, label_fact.get(0), lifted_example)
                else:
                    value = label_fact.get(0).getValue()
                    query_atom = examples_builder.createQueryAtom(
                        str(self.examples_counter), label_fact.get(0), lifted_example
                    )
            else:
                raise NotImplementedError

            if not example_query and examples_queries:
                raise Exception("Inconsistent examples! Some examples have queries and some do not")

            examples_queries = example_query
            logic_samples.append(self.logic_sample(value, query_atom))
            self.examples_counter += 1
        return logic_samples, examples_queries

    def build_dataset(
        self, dataset: Dataset, backend: Backend, settings: SettingsProxy, file_mode: bool = False
    ) -> BuiltDataset:
        """Builds the dataset (does grounding and neuralization) for this template instance and the backend

        :param dataset:
        :param backend:
        :param settings:
        :param file_mode:
        :return:
        """
        self.examples_counter = 0
        self.query_counter = 0

        weight_factory = self.java_factory.weight_factory

        examples_builder = self.examples_builder(settings.settings)
        query_builder = self.queries_builder(settings.settings)
        query_builder.setFactoriesFrom(examples_builder)

        settings.settings.groundingMode = self.grounding_mode.INDEPENDENT

        if not dataset.file_sources:
            examples = dataset.examples
            queries = dataset.queries

            if dataset.data is not None:
                if file_mode:
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as q_tf:
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as e_tf:
                            dataset.dump(q_tf, e_tf)

                            q_tf.flush()
                            e_tf.flush()

                            return self.build_dataset(
                                Dataset(examples_file=e_tf.name, queries_file=q_tf.name), backend, settings, False
                            )
                examples = []
                queries = []

                for data in dataset.data:
                    query, example = data.to_logic_form(
                        one_hot_encode_labels=dataset.one_hot_encode_labels,
                        one_hot_decode_features=dataset.one_hot_decode_features,
                        max_classes=dataset.number_of_classes,
                    )

                    examples.append(example)
                    queries.append(query)

            if len(examples) == 1:
                settings.settings.groundingMode = self.grounding_mode.GLOBAL

            self.java_factory.weight_factory = self.java_factory.get_new_weight_factory()
            examples, example_queries = self.build_examples(examples, examples_builder)

            self.java_factory.weight_factory = self.java_factory.get_new_weight_factory()
            queries, one_query_per_example = self.build_queries(queries, query_builder)

            logic_samples = DatasetBuilder.merge_queries_with_examples(
                queries, examples, one_query_per_example, example_queries
            )
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
    def merge_queries_with_examples(queries, examples, one_query_per_example, example_queries=True):
        if len(examples) == 0:
            return queries

        if len(queries) == 0:
            if not example_queries:
                raise Exception("No queries provided! The query list is empty and examples do not contain queries")
            return examples

        # One large example for one or more queries
        if len(examples) == 1:
            logic_samples = []
            for query in queries:
                example = examples[0] if query.query.evidence is None else query
                query_object = query if query.isQueryOnly else examples[0]
                query_object.query.evidence = example.query.evidence
                logic_samples.append(query)
            return logic_samples

        # One example per query
        if one_query_per_example:
            if len(examples) != len(queries):
                raise Exception(
                    f"The size of examples {len(examples)} doesn't match the size of queries {len(queries)}"
                )

            logic_samples = []

            for query, example in zip(queries, examples):
                example_object = example if query.query.evidence is None else query
                query_object = query if query.isQueryOnly else example
                query_object.query.evidence = example_object.query.evidence
                logic_samples.append(query)
            return logic_samples

        logic_samples = []
        example_map = {e.getId(): e for e in examples}

        # Multiple queries per example
        for query in queries:
            query_id = query.getId()

            example_object = example_map[query_id] if query.query.evidence is None else query
            query_object = query if query.isQueryOnly else example_map[query_id]
            query_object.query.evidence = example_object.query.evidence
            logic_samples.append(query)
        return logic_samples
