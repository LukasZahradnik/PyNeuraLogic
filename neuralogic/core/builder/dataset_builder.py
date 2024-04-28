from typing import Union, Set, Dict, List

import jpype

import neuralogic.dataset as datasets
from neuralogic import is_initialized, initialize
from neuralogic.core.builder.builder import Builder
from neuralogic.core.builder.components import BuiltDataset, GroundedDataset
from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.core.constructs.rule import Rule
from neuralogic.core.constructs.java_objects import JavaFactory
from neuralogic.core.settings import SettingsProxy
from neuralogic.core.sources import Sources

TemplateEntries = Union[BaseRelation, WeightedRelation, Rule]


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
            head, facts = self.java_factory.get_query(query)

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

    def build_examples(self, examples, examples_builder, learnable_facts=False):
        logic_samples = []
        one = jpype.JClass("cz.cvut.fel.ida.algebra.values.ScalarValue")(1.0)
        examples_queries = False

        for example in examples:
            if example is None:
                example = []

            label, lifted_example = self.java_factory.get_lifted_example(example, learnable_facts)
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

    def ground_dataset(
        self,
        dataset: datasets.BaseDataset,
        settings: SettingsProxy,
        *,
        batch_size: int = 1,
        learnable_facts: bool = False,
    ) -> GroundedDataset:
        """Grounds the dataset

        :param dataset:
        :param settings:
        :param batch_size:
        :param learnable_facts:
        :return:
        """
        if isinstance(dataset, datasets.ConvertibleDataset):
            return self.ground_dataset(
                dataset.to_dataset(),
                settings,
                batch_size=batch_size,
                learnable_facts=learnable_facts,
            )

        if batch_size > 1:
            settings.settings.minibatchSize = batch_size
            settings.settings.parallelTraining = True

        builder = Builder(settings)

        if isinstance(dataset, datasets.Dataset):
            self.examples_counter = 0
            self.query_counter = 0

            weight_factory = self.java_factory.weight_factory

            examples_builder = self.examples_builder(settings.settings)
            query_builder = self.queries_builder(settings.settings)
            query_builder.setFactoriesFrom(examples_builder)

            settings.settings.groundingMode = self.grounding_mode.INDEPENDENT
            if len(dataset.samples) != 0 and (len(dataset._examples) != 0 or len(dataset._queries) != 0):
                raise ValueError("Cannot provide both samples and examples with queries")

            examples = dataset._examples
            queries = dataset._queries

            if len(dataset.samples) != 0:
                examples, queries = self.samples_to_examples_and_queries(dataset.samples)

            if len(examples) == 1:
                settings.settings.groundingMode = self.grounding_mode.GLOBAL
            settings.settings.infer()
            settings._setup_random_generator()

            self.java_factory.weight_factory = self.java_factory.get_new_weight_factory()
            examples, example_queries = self.build_examples(examples, examples_builder, learnable_facts)

            self.java_factory.weight_factory = self.java_factory.get_new_weight_factory()
            queries, one_query_per_example = self.build_queries(queries, query_builder)

            logic_samples = DatasetBuilder.merge_queries_with_examples(
                queries, examples, one_query_per_example, example_queries
            )

            groundings = builder.ground_from_logic_samples(self.parsed_template, logic_samples)

            self.java_factory.weight_factory = weight_factory
        elif isinstance(dataset, datasets.FileDataset):
            args = ["-t", dataset.examples_file or dataset.queries_file]
            if dataset.queries_file is not None:
                args.extend(["-q", dataset.queries_file])
            if dataset.examples_file is not None:
                args.extend(["-e", dataset.examples_file])
            sources = Sources.from_args(args, settings)

            groundings = builder.ground_from_sources(self.parsed_template, sources)
        else:
            raise NotImplementedError

        return GroundedDataset(groundings, builder)

    def build_dataset(
        self,
        dataset: Union[datasets.BaseDataset, GroundedDataset],
        settings: SettingsProxy,
        *,
        batch_size: int = 1,
        learnable_facts: bool = False,
        progress: bool = False,
    ) -> BuiltDataset:
        """Builds the dataset (does grounding and neuralization)

        :param dataset:
        :param settings:
        :param batch_size:
        :param learnable_facts:
        :param progress:
        :return:
        """
        grounded_dataset = dataset

        if not isinstance(dataset, GroundedDataset):
            grounded_dataset = self.ground_dataset(
                dataset, settings, batch_size=batch_size, learnable_facts=learnable_facts
            )
        return BuiltDataset(grounded_dataset.neuralize(progress), batch_size)

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

    @staticmethod
    def samples_to_examples_and_queries(samples: List):
        example_dict = {}
        queries_dict = {}

        for sample in samples:
            idx = id(sample.example)

            if idx not in example_dict:
                queries_dict[idx] = [sample.query]
                example_dict[idx] = sample.example
            else:
                queries_dict[idx].append(sample.query)
        return example_dict.values(), queries_dict.values()
