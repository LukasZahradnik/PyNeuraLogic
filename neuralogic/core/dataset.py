from typing import Optional, List, Union, Tuple, Sequence, Iterable

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
        y: Union[Sequence, float, int] = None,
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.y_mask = y_mask

    @staticmethod
    def get_query(
        y, output_name: str = "predict", one_hot_encode_labels: bool = False, max_classes=1, index: Optional[int] = None
    ):
        relation = Relation.get(output_name)

        if index is not None:
            relation = relation(index)

        if one_hot_encode_labels:
            vector = y

            if not isinstance(y, (Sequence, np.ndarray)) or len(y) != max_classes:
                vector = np.zeros((max_classes,))
                vector[y[0] if isinstance(y, (Sequence, np.ndarray)) else y] = 1
            query = relation[vector]
        elif not isinstance(y, Iterable) or (getattr(y, "shape", -1) in (1, tuple())):
            query = relation[float(y)]
        elif len(y) == 1 and not isinstance(y[0], Sequence):
            query = relation[float(y[0])]
        else:
            if isinstance(y, (Sequence, np.ndarray)):
                query = relation[y]
            else:
                query = relation[y.detach().numpy()]
        return query

    def to_logic_form(
        self,
        feature_name: str = "node_feature",
        edge_name: str = "edge",
        output_name: str = "predict",
        one_hot_encode_labels=False,
        one_hot_decode_features=False,
        max_classes=1,
    ) -> Tuple:
        if self.y_mask is not None:
            query = []

            for i in self.y_mask:
                query.append(Data.get_query(self.y[int(i)], output_name, one_hot_encode_labels, max_classes, int(i)))
        else:
            query = Data.get_query(self.y, output_name, one_hot_encode_labels, max_classes)

        if self.edge_attr is None:
            example = [
                Relation.get(edge_name)(int(u), int(v))[1].fixed()
                for u, v in zip(self.edge_index[0], self.edge_index[1])
            ]
        elif isinstance(self.edge_attr, np.ndarray):
            example = [
                Relation.get(edge_name)(int(u), int(v))[w if w.size == 1 else w].fixed()
                for u, v, w in zip(self.edge_index[0], self.edge_index[1], self.edge_attr)
            ]
        elif isinstance(self.edge_attr, (Sequence, np.ndarray)):
            example = [
                Relation.get(edge_name)(int(u), int(v))[
                    w if len(w) == 1 and isinstance(w[0], float, int) else w
                ].fixed()
                for u, v, w in zip(self.edge_index[0], self.edge_index[1], self.edge_attr)
            ]
        else:
            example = [
                Relation.get(edge_name)(int(u), int(v))[w if w.size == 1 else w].fixed()
                for u, v, w in zip(self.edge_index[0], self.edge_index[1], self.edge_attr.detach().numpy())
            ]

        if one_hot_decode_features:
            if isinstance(self.x, (list, np.ndarray)):
                for i, features in enumerate(self.x):
                    class_ = np.argmax(features)
                    example.append(Relation.get(f"{feature_name}_{class_}")(i)[1].fixed())
            else:
                for i, features in enumerate(self.x):
                    class_ = np.argmax(features)
                    example.append(Relation.get(f"{feature_name}_{class_}")(i)[1].fixed())
        else:
            if isinstance(self.x, np.ndarray):
                for i, features in enumerate(self.x):
                    example.append(
                        Relation.get(feature_name)(i)[features[0] if features.size == 1 else features].fixed()
                    )
            elif isinstance(self.x, list):
                for i, features in enumerate(self.x):
                    example.append(
                        Relation.get(feature_name)(i)[
                            features[0] if len(features) == 1 and isinstance(features[0], (int, float)) else features
                        ].fixed()
                    )
            else:
                for i, features in enumerate(self.x):
                    weight = features.detach().numpy()
                    example.append(Relation.get(feature_name)(i)[weight[0] if weight.size == 1 else weight].fixed())
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
        one_hot_encode_labels: bool = False,
        one_hot_decode_features: bool = False,
        number_of_classes: int = 1,
        feature_name: str = "node_feature",
        edge_name: str = "edge",
        output_name: str = "predict",
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

        self.one_hot_decode_features = one_hot_decode_features
        self.one_hot_encode_labels = one_hot_encode_labels
        self.number_of_classes = number_of_classes

        self.feature_name: str = feature_name
        self.edge_name: str = edge_name
        self.output_name: str = output_name

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
        sep: str = "\n",
    ):
        for data in self.data:
            query, examples = data.to_logic_form(
                self.feature_name,
                self.edge_name,
                self.output_name,
                self.one_hot_encode_labels,
                self.one_hot_decode_features,
                self.number_of_classes,
            )

            queries_fp.write(f"{query}{sep}")
            examples_fp.write(f"{','.join(example.to_str(False) for example in examples)}.{sep}")

    def dump_to_file(
        self,
        queries_filename: str,
        examples_filename: str,
        sep: str = "\n",
    ):
        with open(queries_filename, "w") as queries_fp:
            with open(examples_filename, "w") as examples_fp:
                self.dump(queries_fp, examples_fp, sep)
