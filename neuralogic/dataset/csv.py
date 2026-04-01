import enum
from pathlib import Path
from typing import List, Union, TextIO, Callable, Sequence

from neuralogic.core.constructs.factories import R
from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.core.constructs.rule import Rule
from neuralogic.dataset import Dataset, Sample
from neuralogic.dataset.base import ConvertibleDataset

DatasetEntries = Union[BaseRelation, Rule]


class Mode(enum.Enum):
    """
    Enum representing different modes of creating samples from CSV files.
    """
    ONE_EXAMPLE = "one"
    EXAMPLE_PER_SOURCE = "example_per_source"
    ZIP = "zip"


class CSVFile:
    """
    Represents a single CSV file source and its configuration for conversion to logic relations.
    """
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
        value_column: Union[str, int] | None = None,
        default_value: Union[float, int] | None = None,
        value_mapper: Callable | None = None,
        term_columns: Sequence[Union[str, int]] | None = None,
        header: bool = False,
        skip_rows: int = 0,
        n_rows: int | None = None,
        replace_empty_column: Union[str, float, int] = 0,
    ):
        """
        Parameters
        ----------
        relation_name : str
            The name of the relation to create from the CSV rows.
        csv_source : Union[TextIO, Path]
            The source of the CSV data.
        sep : str, optional
            The separator used in the CSV. Default: ",".
        value_column : Union[str, int], optional
            The column containing the relation value (weight). Default: None.
        default_value : Union[float, int], optional
            The default value if not found in CSV. Default: None.
        value_mapper : Callable, optional
            A function to map the CSV value to a different value. Default: None.
        term_columns : Sequence[Union[str, int]], optional
            The columns to use as terms for the relation. Default: None (all columns).
        header : bool, optional
            Whether the CSV file has a header. Default: False.
        skip_rows : int, optional
            The number of rows to skip at the beginning. Default: 0.
        n_rows : int, optional
            The maximum number of rows to read. Default: None (all).
        replace_empty_column : Union[str, float, int], optional
            The value to use for empty columns. Default: 0.
        """
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

    def _get_column_indices(self, header) -> List[int] | None:
        if self.term_columns is None:
            return None
        new_columns = []

        for col_value in self.term_columns:
            new_columns.append(CSVFile._find_index_in_header(header, col_value))
        return new_columns

    def _to_logic(self, fp: TextIO) -> list[DatasetEntries]:
        example: list[DatasetEntries] = []

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
            headers = header.strip().split(self.sep)

            value_column = None if value_column is None else CSVFile._find_index_in_header(headers, value_column)
            use_columns = self._get_column_indices(headers)

        for _ in range(self.skip_rows):
            fp.readline()

        while True:
            line = fp.readline()

            if not line or not line.strip():
                break

            terms = line.strip().split(self.sep)
            if use_columns is None:
                line_relation = relation(
                    [(term.strip().lower() if len(term.strip()) else replace_empty) for term in terms]
                )
            else:
                line_relation = relation(
                    [(terms[i].strip().lower() if len(terms[i].strip()) else replace_empty) for i in use_columns]
                )

            if value_column is None:
                if default_value is not None:
                    line_relation = line_relation[float(default_value)]
            else:
                value = terms[value_column].strip()
                if not len(value):
                    line_value = default_value if default_value is not None else replace_empty
                else:
                    line_value = value

                if value_mapper is None:
                    line_relation = line_relation[float(line_value)]
                else:
                    line_relation = line_relation[value_mapper(line_value)]

            example.append(line_relation)
            read_lines += 1
            if read_lines == self.n_rows:
                break
        return example

    def to_logic_form(self) -> list[DatasetEntries]:
        """
        Converts the CSV source to a list of logic relations.

        Returns
        -------
        list[DatasetEntries]
            The list of created logic relations.
        """
        if isinstance(self.csv_source, (str, Path)):
            with open(self.csv_source, "r") as fp:
                return self._to_logic(fp)
        return self._to_logic(self.csv_source)


class CSVDataset(ConvertibleDataset):
    """
    Represents a dataset composed of one or more CSV files.
    """
    def __init__(
        self,
        csv_files: Union[List[CSVFile], CSVFile],
        csv_queries: CSVFile | None = None,
        mode: Mode = Mode.ONE_EXAMPLE,
    ):
        """
        Parameters
        ----------
        csv_files : Union[List[CSVFile], CSVFile]
            The CSV file(s) containing the examples.
        csv_queries : CSVFile, optional
            The CSV file containing the queries. Default: None.
        mode : Mode, optional
            The mode of creating samples. Default: Mode.ONE_EXAMPLE.
        """
        self.csv_queries = csv_queries
        self.csv_files = [csv_files] if isinstance(csv_files, CSVFile) else csv_files
        self.mode = mode

    def add_csv_file(self, file: CSVFile):
        self.csv_files.append(file)

    def set_query_csv_file(self, file: CSVFile):
        self.csv_queries = file

    def to_dataset(self) -> Dataset:
        """
        Converts the CSV files to a Dataset object.

        Returns
        -------
        Dataset
            The created Dataset object.
        """
        queries: List[BaseRelation | Rule] = self.csv_queries.to_logic_form() if self.csv_queries else []

        if self.mode == Mode.ONE_EXAMPLE:
            example: List[DatasetEntries] = []

            for source in self.csv_files:
                example.extend(source.to_logic_form())
            if not queries:
                return Dataset([Sample(None, example)])
            return Dataset([Sample(q, example) for q in queries])
        elif self.mode == Mode.ZIP:
            logic_examples = [source.to_logic_form() for source in self.csv_files]
            if not queries:
                return Dataset([Sample(None, zipped_example) for zipped_example in zip(*logic_examples)])
            return Dataset([Sample(q, zipped_example) for q, zipped_example in zip(queries, zip(*logic_examples))])
        elif self.mode == Mode.EXAMPLE_PER_SOURCE:
            if not queries:
                return Dataset([Sample(None, source.to_logic_form()) for source in self.csv_files])
            return Dataset([Sample(q, source.to_logic_form()) for source, q in zip(self.csv_files, queries)])
        raise NotImplementedError
