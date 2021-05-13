from typing import Optional, List, Union, Sized

from neuralogic.core.constructs.atom import BaseAtom, WeightedAtom
from neuralogic.core.constructs.rule import Rule


DatasetEntries = Union[BaseAtom, WeightedAtom, Rule]


class Dataset:
    def __init__(
        self,
        *,
        y: Optional[Sized] = None,
        x: Optional[Sized] = None,
        examples_file: Optional[str] = None,
        queries_file: Optional[str] = None,
    ):
        if (y is None and x is not None) or (x is None and y is not None):
            raise Exception

        self.raw_values = False
        self.file_sources = False

        if y is not None and x is not None:
            self.raw_values = True
            if len(y) != len(x):
                raise Exception

        if examples_file is not None or queries_file is not None:
            if self.raw_values:
                raise Exception
            self.file_sources = True

        self.y = y
        self.x = x

        self.examples_file = examples_file
        self.queries_file = queries_file

        self.examples: List[DatasetEntries] = []
        self.queries: List[DatasetEntries] = []

    def __len__(self):
        if self.raw_values:
            return len(self.y)
        if self.file_sources:
            return -1
        return max(len(self.examples), len(self.queries))

    def __getitem__(self, item):
        pass

    def add_example(self, example):
        self.add_examples([example])

    def add_examples(self, examples: List):
        if self.file_sources or self.raw_values:
            raise Exception
        self.examples.extend(examples)

    def set_examples(self, examples: List):
        if self.file_sources or self.raw_values:
            raise Exception
        self.examples = examples

    def add_query(self, query):
        self.add_queries([query])

    def add_queries(self, queries: List):
        if self.file_sources or self.raw_values:
            raise Exception
        self.queries.extend(queries)

    def set_queries(self, queries: List):
        if self.file_sources or self.raw_values:
            raise Exception
        self.queries = queries
