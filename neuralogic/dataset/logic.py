from typing import Optional, List, Union, Sequence

from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.core.constructs.rule import Rule
from neuralogic.dataset.base import BaseDataset

DatasetEntries = Union[BaseRelation, Rule]


class Sample:
    __slots__ = (
        "query",
        "example",
    )

    def __init__(
        self, query: Optional[BaseRelation], example: Optional[Union[Sequence[DatasetEntries], DatasetEntries]]
    ):
        self.query = query

        if example is None:
            example = []

        if not isinstance(example, Sequence):
            self.example = [example]
        else:
            self.example = example

    def __str__(self) -> str:
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

    def __init__(self, samples: Optional[Union[List[Sample], Sample]] = None):
        self.samples = samples

        if self.samples is None:
            self.samples = []
        elif not isinstance(self.samples, list):
            self.samples = [self.samples]

        self._examples = []
        self._queries = []

    def set_samples(self, samples: List[Sample]):
        self.samples = samples

    def add_samples(self, samples: List[Sample]):
        self.samples.extend(samples)

    def add_sample(self, sample: Sample):
        self.samples.append(sample)

    def add(self, query: BaseRelation, example: Optional[List[DatasetEntries]]):
        self.samples.append(Sample(query, example))

    def __getitem__(self, item: int) -> Sample:
        return self.samples[item]

    def __setitem__(self, key: int, value: Sample):
        self.samples[key] = value

    def __delitem__(self, key: int):
        del self.samples

    def __str__(self):
        return ". ".join(str(s) for s in self.samples)

    def __len__(self):
        return len(self.samples)

    # Deprecated
    def add_example(self, example):
        self.add_examples([example])

    def add_examples(self, examples: List):
        self._examples.extend(examples)

    def add_query(self, query):
        self.add_queries([query])

    def add_queries(self, queries: List):
        self._queries.extend(queries)

    def set_examples(self, examples: List):
        self._examples = examples

    def set_queries(self, queries: List):
        self._queries = queries
