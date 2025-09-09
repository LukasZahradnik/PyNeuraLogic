from typing import Union, Sequence

import jpype

from neuralogic.core.constructs.java_objects import JavaFactory
from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.core.constructs.rule import Rule
from neuralogic.dataset.base import BaseDataset
from neuralogic.core.constructs.factories import R

DatasetEntries = Union[BaseRelation, Rule]


class Sample:
    __slots__ = (
        "query",
        "example",
    )

    def __init__(self, query: BaseRelation | list[BaseRelation] | None, example: Sequence[DatasetEntries] | DatasetEntries | None):
        self.query = query

        if example is None:
            example = []

        if not isinstance(example, Sequence):
            self.example = [example]
        else:
            self.example = example

    def draw(*args, **kwargs):
        raise NotImplementedError("sample cannot be drawn unless it is grounded or neuralized")

    def __str__(self) -> str:
        if isinstance(self.query, list):
            return ", ".join(str(q) for q in self.query)
        return str(self.query)

    def __len__(self) -> int:
        if self.example is None:
            return 0
        return len(self.example)


class Dataset(BaseDataset):
    r"""
    Dataset encapsulating (learning) samples in the form of logic format, allowing users to fully take advantage of the
    PyNeuraLogic library.
    """

    __slots__ = ("samples", "_examples", "_queries")

    def __init__(self, samples: list[Sample] | Sample | None = None):
        self.samples = []

        if isinstance(samples, list):
            self.samples = samples
        elif not isinstance(samples, list) and samples is not None:
            self.samples = [samples]

        self._examples: list[list[DatasetEntries]] = []
        self._queries: list[BaseRelation] = []

    def set_samples(self, samples: list[Sample]):
        self.samples = samples

    def add_samples(self, samples: list[Sample]) -> "Dataset":
        self.samples.extend(samples)

        return self

    def add_sample(self, sample: Sample) -> "Dataset":
        self.samples.append(sample)

        return self

    def add(self, query: BaseRelation | list[BaseRelation] | None, example: list[DatasetEntries] | None) -> "Dataset":
        self.samples.append(Sample(query, example))

        return self

    def __getitem__(self, item: int) -> Sample:
        return self.samples[item]

    def __setitem__(self, key: int, value: Sample):
        self.samples[key] = value

    def __delitem__(self, key: int):
        del self.samples[key]

    def __str__(self):
        return ". ".join(str(s) for s in self.samples)

    def __len__(self):
        return len(self.samples)

    # Deprecated
    def add_example(self, example):
        self.add_examples([example])

    def add_examples(self, examples: list):
        self._examples.extend(examples)

    def add_query(self, query):
        self.add_queries([query])

    def add_queries(self, queries: list):
        self._queries.extend(queries)

    def set_examples(self, examples: list):
        self._examples = examples

    def set_queries(self, queries: list):
        self._queries = queries

    def generate_features(self, feature_depth: int = 1, count_groundings: bool = True):
        java_factory = JavaFactory()

        clauses = []
        vertex_lit = R.get("__vert")
        vertex_lit.predicate.special = False
        vertex_lit.predicate.hidden = False

        for sample in self.samples:
            vertex = set()

            for e in sample.example:
                if isinstance(e, Rule):
                    vertex.update(self._get_constants(e.head))

                    for rel in e.body:
                        vertex.update(self._get_constants(rel))
                if isinstance(e, BaseRelation):
                    vertex.update(self._get_constants(e))

            example = [vertex_lit(vert) for vert in vertex]
            example.extend(sample.example)

            clauses.append(java_factory.to_clause(example))

        clause = jpype.java.util.ArrayList(clauses)

        namespace = "cz.cvut.fel.ida.logic.features.generation"

        jpype.JClass(f"{namespace}.FeatureGenerationSettings").COUNT_GROUNDINGS = count_groundings
        features = jpype.JClass(f"{namespace}.FeatureGenerator").generateFeatures(clause, feature_depth)

        table = [[int(i) for i in feats] for feats in features.table]
        clauses = [str(clause) for clause in features.features]

        return table, clauses

    @staticmethod
    def _get_constants(relation: BaseRelation):
        return [term for term in relation.terms if not str(relation)[0].isupper()]
