from typing import Optional, List, Union, Sized

from neuralogic.core.constructs.atom import BaseAtom, WeightedAtom
from neuralogic.core.constructs.rule import Rule


DatasetEntries = Union[BaseAtom, WeightedAtom, Rule]


class Data:
    """
    Stores a learning example in the form of a tensor numeric representation instead of a rule based representation.
    """
    def __init__(
        self,
        x: Sized = None,
        edge_index: Sized = None,
        edge_attr: Optional[Sized] = None,
        y_mask: Sized = None,
        y: Sized = None,
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.y_mask = y_mask

    @staticmethod
    def from_pyg(data) -> List["Data"]:
        """
        Converts a PyTorch Geometric Data instance into a list of PyNeuraLogic :py:class:`~Data` instances.
        The conversion supports :code:`train_mask`, :code:`test_mask` and :code:`val_mask` attributes -
        for each mask the conversion yields a new data instance.

        :param data: The PyTorch Geometric Data instance
        :return: The list of PyNeuraLogic Data instances
        """
        data_list = []

        if hasattr(data, "train_mask"):
            data_list.append(Data(data.x, data.edge_index, data.edge_attr, data.train_mask, data.y))
        if hasattr(data, "test_mask"):
            data_list.append(Data(data.x, data.edge_index, data.edge_attr, data.test_mask, data.y))
        if hasattr(data, "val_mask"):
            data_list.append(Data(data.x, data.edge_index, data.edge_attr, data.val_mask, data.y))
        if len(data_list) == 0:
            data_list.append(Data(data.x, data.edge_index, data.edge_attr, None, data.y))

        return data_list


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
