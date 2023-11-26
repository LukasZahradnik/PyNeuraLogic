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

    One learning sample consists of:

    * Example: A list of logic facts and rules representing some instance (e.g., a graph)

    * Query: A logic fact to mark the output of a model and optionally target label.

    Examples and queries in the dataset can be paired in the following ways:

    * :math:`N:N` - Dataset contains :math:`N` examples and :math:`N` queries. They will be paired by their index.

    .. code:: python

        dataset.add_example(first_example)
        dataset.add_example(second_example)

        dataset.add_query(first_query)
        dataset.add_query(second_query)

        # Learning samples: [first_example, first_query], [second_example, second_query]

    * :math:`1:N` - Dataset contains :math:`1` example and :math:`N` queries. All queries will be run on the example.

    .. code:: python

        dataset.add_example(example)

        dataset.add_query(first_query)
        dataset.add_query(second_query)

        # Learning samples: [example, first_query], [example, second_query]

    * :math:`N:M` - Dataset contains :math:`N` examples and :math:`M` queries (:math:`N \leq M`).
      It pairs queries similarly to the :math:`N:N` case but also allows running multiple queries on a specific example
      (by inserting a list of queries instead of one query).

    .. code:: python

        dataset.add_example(first_example)
        dataset.add_example(second_example)

        dataset.add_query([first_query_0, first_query_1])
        dataset.add_query(second_query)

        # Learning samples:
        #   [first_example, first_query_0], [first_example, first_query_1], [second_example, second_query]

    Parameters
    ----------

    examples : Optional[List]
        List of examples. Default: ``None``
    queries : Optional[List]
        List of queries. Default: ``None``

    """

    __slots__ = ("samples",)

    def __init__(self, samples: Optional[Union[List[Sample], Sample]] = None):
        self.samples = samples

        if self.samples is None:
            self.samples = []
        elif not isinstance(self.samples, list):
            self.samples = [self.samples]

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
