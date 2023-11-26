import csv
import io
from typing import Optional, List, Union, Callable

from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.core.constructs.rule import Rule
from neuralogic.dataset.logic import Dataset
from neuralogic.dataset.csv import CSVDataset, CSVFile, Mode
from neuralogic.dataset.base import ConvertibleDataset

DatasetEntries = Union[BaseRelation, WeightedRelation, Rule]


class DBSource:
    __slots__ = (
        "relation_name",
        "table_name",
        "term_columns",
        "value_column",
        "default_value",
        "value_mapper",
        "skip_rows",
        "n_rows",
        "replace_empty_column",
        "sep",
    )

    def __init__(
        self,
        relation_name: str,
        table_name: str,
        term_columns: List[str],
        value_column: Optional[str] = None,
        default_value: Union[float, int] = 1.0,
        value_mapper: Optional[Callable] = None,
        skip_rows: int = 0,
        n_rows: Optional[int] = None,
        replace_empty_column: Union[str, float, int] = 0,
        sep=",",
    ):
        self.table_name = table_name
        self.relation_name = relation_name
        self.sep = sep
        self.value_column = value_column
        self.default_value = default_value
        self.value_mapper = value_mapper
        self.term_columns = term_columns
        self.skip_rows = skip_rows
        self.n_rows = n_rows
        self.replace_empty_column = replace_empty_column

        if len(term_columns) == 0:
            raise NotImplementedError("Cannot create DBSource with zero terms")

    def to_csv(self, cursor) -> CSVFile:
        source = io.StringIO()

        columns = [term for term in self.term_columns]
        term_columns = list(range(len(columns)))
        value_column = None

        if self.value_column is not None:
            columns.append(self.value_column)
            value_column = len(columns) - 1

        if hasattr(cursor, "copy_to"):
            cursor.copy_to(source, self.table_name, sep=self.sep, null="", columns=columns)
        else:
            cursor.execute(f"SELECT {','.join(columns)} FROM {self.table_name}")
            results = cursor.fetchall()

            csv_writer = csv.writer(source, lineterminator="\n")
            csv_writer.writerows(results)

        source.seek(0)

        return CSVFile(
            self.relation_name,
            source,
            self.sep,
            value_column,
            self.default_value,
            self.value_mapper,
            term_columns,
            False,
            self.skip_rows,
            self.n_rows,
            self.replace_empty_column,
        )


class DBDataset(ConvertibleDataset):
    def __init__(
        self,
        connection,
        db_sources: Union[List[DBSource], DBSource],
        queries_db_source: Optional[DBSource] = None,
        mode: Mode = Mode.ONE_EXAMPLE,
    ):
        self.connection = connection
        self.db_sources = [db_sources] if isinstance(db_sources, DBSource) else db_sources
        self.queries_db_source = queries_db_source
        self.mode = mode

    def add_db_source(self, db_source: DBSource):
        self.db_sources.append(db_source)

    def set_queries(self, db_source: DBSource):
        self.queries_db_source = db_source

    def to_dataset(self) -> Dataset:
        with self.connection.cursor() as cur:
            csv_files = [db_source.to_csv(cur) for db_source in self.db_sources]
            csv_queries = None if self.queries_db_source is None else self.queries_db_source.to_csv(cur)

        csv_dataset = CSVDataset(csv_files, csv_queries, self.mode)

        return csv_dataset.to_dataset()
