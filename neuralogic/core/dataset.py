from typing import Optional, List, Union, Tuple, Sequence

import numpy as np

from neuralogic.core.constructs.factories import Relation
from neuralogic.core.constructs.atom import BaseAtom, WeightedAtom
from neuralogic.core.constructs.rule import Rule


DatasetEntries = Union[BaseAtom, WeightedAtom, Rule]


class Data:
    """
    Stores a learning example in the form of a tensor numeric representation instead of a rule based representation.
    """

    def __init__(
        self,
        x: Sequence = None,
        edge_index: Sequence = None,
        edge_attr: Optional[Sequence] = None,
        y_mask: Sequence = None,
        y: Sequence = None,
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.y_mask = y_mask

    def to_logic_form(
        self,
        feature_name: str = "node_feature",
        edge_name: str = "edge",
        output_name: str = "predict",
        one_hot_encoding=False,
        max_classes=1,
    ) -> Tuple:
        if one_hot_encoding:
            vector = self.y
            if len(self.y) != max_classes:
                vector = np.zeros((max_classes,))
                vector[self.y[0]] = 1
            query = Relation.get(output_name)[vector]
        elif len(self.y) == 1 and not isinstance(self.y[0], Sequence):
            query = Relation.get(output_name)[float(self.y[0])]
        else:
            if isinstance(self.y, (list, np.ndarray)):
                query = Relation.get(output_name)[self.y]
            else:
                query = Relation.get(output_name)[self.y.detach().numpy()]

        example = [
            Relation.get(edge_name)(int(u), int(v))[1].fixed() for u, v in zip(self.edge_index[0], self.edge_index[1])
        ]

        if isinstance(self.x, (list, np.ndarray)):
            for i, features in enumerate(self.x):
                example.append(Relation.get(feature_name)(i)[features.detach().numpy()].fixed())
        else:
            for i, features in enumerate(self.x):
                example.append(Relation.get(feature_name)(i)[features.detach().numpy()].fixed())
        return query, example

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
        one_hot_encoding: bool = False,
        number_of_classes: int = 1,
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

        self.one_hot_encoding = one_hot_encoding
        self.number_of_classes = number_of_classes

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

    def dump(
        self,
        queries_fp,
        examples_fp,
        feature_name: str = "node_feature",
        edge_name: str = "edge",
        output_name: str = "predict",
        sep: str = "\n",
    ):
        for data in self.data:
            query, examples = data.to_logic_form(
                feature_name, edge_name, output_name, self.one_hot_encoding, self.number_of_classes
            )

            queries_fp.write(f"{query}{sep}")
            examples_fp.write(f"{','.join(example.to_str(False) for example in examples)}.{sep}")
