from typing import Any, Dict, List, Tuple

from ..create_table_parser import parse_create_table
from ..data.query_data import QueryData
from ..insert_query_parser import parse_insert_query
from .database import Database


class QueryNotSupportedException(Exception):
    pass


class IncorrectQueryException(Exception):
    pass


class PythonDatabase(Database):
    columns: Dict[str, List[Tuple[str, str]]] = {}
    values: Dict[str, List[List[Any]]] = {}

    def execute_query(
        self, query: str, params: Tuple[str, ...] = ()
    ) -> List[Tuple[Any, ...]]:
        query_upper = query.upper()
        if query_upper.startswith("INSERT INTO"):
            query_data = QueryData(query, {})
            parse_insert_query(query_data)

            if not query_data.table:
                raise IncorrectQueryException()

            if not query_data.columns and len(query_data.values[0]) != len(
                self.columns[query_data.table]
            ):
                raise IncorrectQueryException()
            elif not query_data.columns:
                query_data.columns = [
                    column[0] for column in self.get_all_columns(query_data.table)
                ]

            if any([len(row) != len(query_data.columns) for row in query_data.values]):
                raise IncorrectQueryException()

            self.insert(query_data)

        elif query_upper.startswith("CREATE TABLE"):
            table_name, column_names, column_types = parse_create_table(query)
            self.create_table(table_name, column_names, column_types)
        else:
            raise QueryNotSupportedException()

    def select_all_data(self, table_name: str) -> List[Tuple[Any, ...]]:
        return [tuple(row) for row in self.values[table_name]]

    def get_all_tables(self) -> List[str]:
        return self.columns.keys()

    def get_all_columns(self, table_name: str) -> List[Tuple[str, str]]:
        return self.columns[table_name]

    def get_example_rows(self, table_name: str) -> List[Tuple[Any, ...]]:
        return [tuple(row) for row in self.values[table_name][:3]]

    def get_database_state(self) -> Dict[str, Tuple[List[str], List[Tuple[Any, ...]]]]:
        return {
            table: (
                [column[0] for column in self.get_all_columns(table)],
                self.get_example_rows(table),
            )
            for table in self.get_all_tables()
        }

    def create_table(
        self, table_name: str, column_names: List[str], column_types: List[str]
    ) -> None:
        self.columns[table_name] = list(zip(column_names, column_types))
        self.values[table_name] = []

    def create_column(
        self, table_name: str, column_name: str, column_type: str
    ) -> None:
        self.columns[table_name].append((column_name, column_type))
        for row in self.values[table_name]:
            row.append(None)

    def remove_table(self, table_name: str) -> None:
        del self.columns[table_name]
        del self.values[table_name]

    def reset_database(self) -> None:
        self.columns = {}
        self.values = {}

    def insert(self, query_data: QueryData) -> None:
        """Inserts the data into the database"""
        for query_row in query_data.values:
            database_row = [None for column in self.columns[query_data.table]]

            for query_column, query_colummn_type, value in zip(
                query_data.columns, query_data.column_types, query_row
            ):
                column_index = [
                    i
                    for i, column in enumerate(self.columns[query_data.table])
                    if column[0] == query_column
                ]
                if len(column_index) != 1:
                    raise IncorrectQueryException()

                database_row[column_index[0]] = (
                    None
                    if value == "NULL"
                    else (
                        int(value)
                        if query_colummn_type == "BIGINT"
                        else (
                            float(value)
                            if query_colummn_type == "DOUBLE"
                            else value[1 : len(value) - 1]
                        )
                    )
                )

            self.values[query_data.table].append(database_row)
