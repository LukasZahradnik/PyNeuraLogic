import enum
from pathlib import Path
from typing import Optional, List, Union, TextIO, Callable

from neuralogic.core.constructs.factories import R
from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.core.constructs.rule import Rule
from neuralogic.dataset import Dataset
from neuralogic.dataset.base import ConvertableDataset

DatasetEntries = Union[BaseRelation, WeightedRelation, Rule]


class Mode(enum.Enum):
    ONE_EXAMPLE = "one"
    EXAMPLE_PER_SOURCE = "example_per_source"
    ZIP = "zip"


class CSVFile:
    __slots__ = (
        "relation_name",
        "csv_source",
        "sep",
        "value_column",
        "default_value",
        "value_mapper",
        "term_columns",
        "header",
        "skip_rows",
        "n_rows",
        "replace_empty_column",
    )

    def __init__(
        self,
        relation_name: str,
        csv_source: Union[TextIO, Path],
        sep=",",
        value_column: Optional[Union[str, int]] = None,
        default_value: Optional[Union[float, int]] = None,
        value_mapper: Optional[Callable] = None,
        term_columns: Optional[List[Union[str, int]]] = None,
        header: bool = False,
        skip_rows: int = 0,
        n_rows: Optional[int] = None,
        replace_empty_column: Union[str, float, int] = 0,
    ):
        self.relation_name = relation_name
        self.csv_source = csv_source
        self.sep = sep
        self.value_column = value_column
        self.default_value = default_value
        self.value_mapper = value_mapper
        self.term_columns = term_columns
        self.header = header
        self.skip_rows = skip_rows
        self.n_rows = n_rows
        self.replace_empty_column = replace_empty_column

    @staticmethod
    def _find_index_in_header(header, value) -> int:
        for index, header_value in enumerate(header):
            if value == header_value:
                return index
        raise ValueError(f"Value {value} not found in the header {header}")

    def _get_column_indices(self, header) -> Optional[List[int]]:
        if self.term_columns is None:
            return None
        new_columns = []

        for col_value in self.term_columns:
            new_columns.append(CSVFile._find_index_in_header(header, col_value))
        return new_columns

    def _to_logic(self, fp: TextIO) -> List[DatasetEntries]:
        example = []

        use_columns = self.term_columns
        value_column = self.value_column
        default_value = self.default_value
        value_mapper = self.value_mapper
        relation = R.get(self.relation_name)
        replace_empty = self.replace_empty_column
        read_lines = 0

        if self.header:
            header = fp.readline()
            if not header:
                return example
            header = header.strip().split(self.sep)

            value_column = None if value_column is None else CSVFile._find_index_in_header(header, value_column)
            use_columns = self._get_column_indices(header)

        for _ in range(self.skip_rows):
            fp.readline()

        while True:
            line = fp.readline()

            if not line or not line.strip():
                break

            terms = line.strip().split(self.sep)
            if use_columns is None:
                line_relation = relation([(term.strip() if len(term.strip()) else replace_empty) for term in terms])
            else:
                line_relation = relation(
                    [(terms[i].strip() if len(terms[i].strip()) else replace_empty) for i in use_columns]
                )

            if value_column is None:
                if default_value is not None:
                    line_relation = line_relation[float(default_value)]
            else:
                value = terms[value_column].strip()
                if not len(value):
                    value = default_value if default_value is not None else replace_empty
                if value_mapper is None:
                    line_relation = line_relation[float(value)]
                else:
                    line_relation = line_relation[value_mapper(value)]

            example.append(line_relation)
            read_lines += 1
            if read_lines == self.n_rows:
                break
        return example

    def to_logic_form(self) -> List[DatasetEntries]:
        if isinstance(self.csv_source, (str, Path)):
            with open(self.csv_source, "r") as fp:
                return self._to_logic(fp)
        return self._to_logic(self.csv_source)


class CSVDataset(ConvertableDataset):
    def __init__(
        self,
        csv_files: Union[List[CSVFile], CSVFile],
        csv_queries: Optional[CSVFile] = None,
        mode: Mode = Mode.ONE_EXAMPLE,
    ):
        self.csv_queries = csv_queries
        self.csv_files = [csv_files] if isinstance(csv_files, CSVFile) else csv_files
        self.mode = mode

    def add_csv_file(self, file: CSVFile):
        self.csv_files.append(file)

    def set_query_csv_file(self, file: CSVFile):
        self.csv_queries = file

    def to_dataset(self) -> Dataset:
        examples = []
        queries = self.csv_queries.to_logic_form() if self.csv_queries else []
        dataset = Dataset(examples, queries)

        if self.mode == Mode.ONE_EXAMPLE:
            example = []

            for source in self.csv_files:
                example.extend(source.to_logic_form())
            examples.append(example)
        elif self.mode == Mode.ZIP:
            logic_examples = [source.to_logic_form() for source in self.csv_files]

            for zipped_example in zip(*logic_examples):
                examples.append(zipped_example)
        elif self.mode == Mode.EXAMPLE_PER_SOURCE:
            for source in self.csv_files:
                examples.append(source.to_logic_form())
        return dataset
