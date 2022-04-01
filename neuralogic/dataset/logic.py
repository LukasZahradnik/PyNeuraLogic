from typing import Optional, List, Union

from neuralogic.core.constructs.atom import BaseAtom, WeightedAtom
from neuralogic.core.constructs.rule import Rule
from neuralogic.dataset.base import BaseDataset

DatasetEntries = Union[BaseAtom, WeightedAtom, Rule]


class Dataset(BaseDataset):
    def __init__(self, examples: Optional[List[DatasetEntries]] = None, queries: Optional[List[DatasetEntries]] = None):
        self.examples: List[DatasetEntries] = examples if examples is not None else []
        self.queries: List[DatasetEntries] = queries if queries is not None else []

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
