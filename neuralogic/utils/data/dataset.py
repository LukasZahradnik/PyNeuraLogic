from typing import Optional, List, Union, Sized

from neuralogic.core.constructs.atom import BaseAtom, WeightedAtom
from neuralogic.core.constructs.rule import Rule


DatasetEntries = Union[BaseAtom, WeightedAtom, Rule]


class Data:
    def __init__(
        self,
        x: Sized = None,
        edge_index: Sized = None,
        edge_attr: Optional[Sized] = None,
        y: Sized = None,
    ):
        if (y is None and edge_index is not None) or (edge_index is None and y is not None):
            raise Exception

        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y


class Dataset:
    def __init__(
        self,
        *,
        data: Optional[List[Data]] = None,
        examples_file: Optional[str] = None,
        queries_file: Optional[str] = None,
    ):
        self.file_sources = False

        if examples_file is not None or queries_file is not None:
            if data is not None:
                raise Exception
            self.file_sources = True

        self.data = data
        self.examples_file = examples_file
        self.queries_file = queries_file

        self.examples: List[DatasetEntries] = []
        self.queries: List[DatasetEntries] = []

    def __len__(self):
        if self.data is not None:
            return len(self.data)
        if self.file_sources:
            return -1
        return max(len(self.examples), len(self.queries))

    def __getitem__(self, item):
        pass

    def add_example(self, example):
        self.add_examples([example])

    def add_examples(self, examples: List):
        if self.file_sources or self.data is not None:
            raise Exception
        self.examples.extend(examples)

    def set_examples(self, examples: List):
        if self.file_sources or self.data is not None:
            raise Exception
        self.examples = examples

    def add_query(self, query):
        self.add_queries([query])

    def add_queries(self, queries: List):
        if self.file_sources or self.data is not None:
            raise Exception
        self.queries.extend(queries)

    def set_queries(self, queries: List):
        if self.file_sources or self.data is not None:
            raise Exception
        self.queries = queries
