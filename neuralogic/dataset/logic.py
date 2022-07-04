from typing import Optional, List, Union

from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.core.constructs.rule import Rule
from neuralogic.dataset.base import BaseDataset

DatasetEntries = Union[BaseRelation, WeightedRelation, Rule]


class Dataset(BaseDataset):
    r"""
    Dataset encapsulating (learning) samples in the form of logic format, allowing users to fully take advantage of the
    PyNeuraLogic library.

    One learning sample consists of:
    * Example: A list of logic facts and rules representing some instance (e.g., a graph)
    * Query: A logic fact to mark the output of a model and optionally target label.

    Examples and queries in the dataset can be paired in the following ways:

    * N:N - Dataset contains N examples and N queries. They will be paired by their index.

    .. code:: python

        dataset.add_example(first_example)
        dataset.add_example(second_example)

        dataset.add_query(first_query)
        dataset.add_query(second_query)

        # Learning samples: [first_example, first_query], [second_example, second_query]

    * 1:N - Dataset contains 1 example and N queries. All queries will be run on the example.

    .. code:: python

        dataset.add_example(example)

        dataset.add_query(first_query)
        dataset.add_query(second_query)

        # Learning samples: [example, first_query], [example, second_query]

    * N:M - Dataset contains N examples and M queries (N <= M). It pairs queries similarly to the N: N case but also
      allows running multiple queries on a specific example (by inserting a list of queries instead of one query).

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

    def __init__(
        self,
        examples: Optional[List[List[DatasetEntries]]] = None,
        queries: Optional[List[Union[List[DatasetEntries], DatasetEntries]]] = None,
    ):
        self.examples: List[List[DatasetEntries]] = examples if examples is not None else []
        self.queries: List[Union[List[DatasetEntries], DatasetEntries]] = queries if queries is not None else []

    def add_example(self, example):
        self.add_examples([example])

    def add_examples(self, examples: List):
        self.examples.extend(examples)

    def set_examples(self, examples: List):
        self.examples = examples

    def add_query(self, query):
        self.add_queries([query])

    def add_queries(self, queries: List):
        self.queries.extend(queries)

    def set_queries(self, queries: List):
        self.queries = queries

    def dump(
        self,
        queries_fp,
        examples_fp,
        sep: str = "\n",
    ):
        for examples in self.examples:
            examples_fp.write(f"{','.join(example.to_str(False) for example in examples)}.{sep}")

        for query in self.queries:
            queries_fp.write(f"{query}{sep}")
