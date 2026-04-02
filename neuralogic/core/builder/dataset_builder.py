from collections.abc import Iterable
from typing import Any

import jpype

import neuralogic.dataset as datasets
from neuralogic.setup import is_initialized, initialize
from neuralogic.core.builder.builder import Builder
from neuralogic.core.builder.dataset import BuiltDataset, GroundedDataset
from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.core.constructs.rule import Rule
from neuralogic.core.constructs.java_objects import JavaFactory
from neuralogic.core.settings import SettingsProxy
from neuralogic.core.sources import Sources

ModelEntries = BaseRelation | WeightedRelation | Rule


class DatasetBuilder:
    """
    DatasetBuilder is responsible for grounding and neuralizing datasets.
    """

    def __init__(self, parsed_model: Any, java_factory: JavaFactory):
        """
        Parameters
        ----------
        parsed_model : Any
            The parsed model.
        java_factory : JavaFactory
            The java factory.
        """
        if not is_initialized():
            initialize()

        self.java_factory = java_factory
        self.parsed_model = parsed_model

        self.grounding_mode = jpype.JClass("cz.cvut.fel.ida.setup.Settings").GroundingMode
        self.logic_sample = jpype.JClass("cz.cvut.fel.ida.logic.constructs.example.LogicSample")

        self.examples_builder = jpype.JClass("cz.cvut.fel.ida.logic.constructs.building.ExamplesBuilder")
        self.queries_builder = jpype.JClass("cz.cvut.fel.ida.logic.constructs.building.QueriesBuilder")

        self.query_counter = 0
        self.examples_counter = 0

    def build_queries(self, queries: Iterable[Any], query_builder: Any) -> tuple[list[Any], bool]:
        """
        Builds queries from the provided queries and query builder.

        Parameters
        ----------
        queries : Iterable
            The queries to build.
        query_builder : Any
            The query builder.

        Returns
        -------
        tuple[list[Any], bool]
            A tuple containing the list of built logic samples and a boolean indicating if there is one query per example.
        """
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
            elif facts is not None:
                id = str(self.query_counter)
                if len(facts) > 1:
                    one_query_per_example = False

                logic_samples.extend(
                    [self.logic_sample(f.getValue(), query_builder.createQueryAtom(id, f), True) for f in facts]
                )
            else:
                logic_samples.append(None)
            self.query_counter += 1
        return logic_samples, one_query_per_example

    def build_examples(self, examples: Iterable[Any], examples_builder: Any, learnable_facts: bool = False) -> tuple[list[Any], bool]:
        """
        Builds examples from the provided examples and examples builder.

        Parameters
        ----------
        examples : Iterable
            The examples to build.
        examples_builder : Any
            The examples builder.
        learnable_facts : bool
            Whether facts are learnable. Default: False.

        Returns
        -------
        tuple[list[Any], bool]
            A tuple containing the list of built logic samples and a boolean indicating if there are examples with queries.
        """
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
        progress: bool = False,
        raw_groundings: bool = False,
    ) -> GroundedDataset | Any:
        """Grounds the dataset.

        Parameters
        ----------
        dataset : datasets.BaseDataset
            The dataset to ground.
        settings : SettingsProxy
            The settings proxy.
        batch_size : int
            The batch size. Default: 1.
        learnable_facts : bool
            Whether facts are learnable. Default: False.
        progress : bool
            Whether to show progress. Default: False.
        raw_groundings : bool
            Whether to return raw groundings. Default: False.

        Returns
        -------
        GroundedDataset | Any
            The grounded dataset or raw groundings.
        """
        if isinstance(dataset, datasets.ConvertibleDataset):
            return self.ground_dataset(
                dataset.to_dataset(),
                settings,
                batch_size=batch_size,
                learnable_facts=learnable_facts,
                progress=progress,
                raw_groundings=raw_groundings,
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
            builder.settings.settings.oneQueryPerExample = one_query_per_example

            logic_samples = DatasetBuilder.merge_queries_with_examples(
                queries, examples, one_query_per_example, example_queries
            )

            groundings = builder.ground_from_logic_samples(self.parsed_model, logic_samples, progress)

            self.java_factory.weight_factory = weight_factory
        elif isinstance(dataset, datasets.FileDataset):
            if dataset.queries_file is None and dataset.examples_file is None:
                raise ValueError("To build FileDataset provide either queries or examples")

            args = ["-t", dataset.examples_file or dataset.queries_file or ""]
            if dataset.queries_file is not None:
                args.extend(["-q", dataset.queries_file])
            if dataset.examples_file is not None:
                args.extend(["-e", dataset.examples_file])
            sources = Sources.from_args(args, settings)

            groundings = builder.ground_from_sources(self.parsed_model, sources, progress)
        else:
            raise NotImplementedError

        if raw_groundings:
            return groundings
        if progress:
            return GroundedDataset(groundings, builder)

        return GroundedDataset(groundings.collect(builder.collectors.toList()), builder)

    def build_dataset(
        self,
        dataset: datasets.BaseDataset | GroundedDataset,
        settings: SettingsProxy,
        *,
        batch_size: int = 1,
        learnable_facts: bool = False,
        progress: bool = False,
    ) -> BuiltDataset:
        """Builds the dataset (does grounding and neuralization).

        Parameters
        ----------
        dataset : datasets.BaseDataset | GroundedDataset
            The dataset to build.
        settings : SettingsProxy
            The settings proxy.
        batch_size : int
            The batch size. Default: 1.
        learnable_facts : bool
            Whether facts are learnable. Default: False.
        progress : bool
            Whether to show progress. Default: False.

        Returns
        -------
        BuiltDataset
            The built dataset.
        """
        if not isinstance(dataset, GroundedDataset):
            groundings = self.ground_dataset(
                dataset, settings, batch_size=batch_size, learnable_facts=learnable_facts, raw_groundings=True,
            )

            samples = Builder(settings).neuralize(groundings, progress, None)
            return BuiltDataset(samples, batch_size)
        return dataset.neuralize(batch_size=batch_size, progress=progress)

    @staticmethod
    def merge_queries_with_examples(self, queries: list[Any], examples: list[Any], one_query_per_example: bool, example_queries: bool = True) -> list[Any]:
        """
        Merges queries with their corresponding examples.

        Parameters
        ----------
        queries : list[Any]
            The list of queries.
        examples : list[Any]
            The list of examples.
        one_query_per_example : bool
            Whether there is one query per example.
        example_queries : bool
            Whether examples contain queries. Default: True.

        Returns
        -------
        list[Any]
            The list of merged logic samples.
        """
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
                if query is None:
                    logic_samples.append(examples[0])
                    continue
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
                if query is None:
                    logic_samples.append(example)
                    continue
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

    def samples_to_examples_and_queries(samples: list[Any]) -> tuple[Iterable[Any], Iterable[Any]]:
        """
        Converts a list of samples to two lists: examples and queries.

        Parameters
        ----------
        samples : list[Any]
            The list of samples.

        Returns
        -------
        tuple[Iterable[Any], Iterable[Any]]
            A tuple containing the iterable of examples and the iterable of queries.
        """
        example_dict = {}
        queries_dict = {}

        for sample in samples:
            idx = id(sample.example)

            if idx not in example_dict:
                if isinstance(sample.query, list):
                    queries_dict[idx] = [*sample.query]
                else:
                    queries_dict[idx] = [sample.query]
                example_dict[idx] = sample.example
                continue

            if isinstance(sample.query, list):
                queries_dict[idx].extend(sample.query)
            else:
                queries_dict[idx].append(sample.query)
        return example_dict.values(), queries_dict.values()
