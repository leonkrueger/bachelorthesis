from typing import Any, Dict, List, Tuple

from ..data.query_data import QueryData
from ..data.table_origin import TableOrigin
from ..insert_query_parser import parse_insert_query
from .database import Database


class QueryNotSupportedException(Exception):
    pass


class IncorrectQueryException(Exception):
    pass


class PythonDatabase(Database):
    columns: Dict[str, List[Tuple[str, str]]] = {}
    values: Dict[str, List[List[Any]]] = {}

    def __init__(self) -> None:
        # All tables used for the internal database management
        enum_values_string = {
            ", ".join([f"'{table_origin.value}'" for table_origin in TableOrigin])
        }
        self.internal_tables = {
            "table_registry": (
                ["name", "name_origin"],
                ["VARCHAR(1023)", f"ENUM({enum_values_string})"],
            )
        }

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
                query_data.columns = self.get_all_columns(query_data.table)

            if any([len(row) != len(query_data.columns) for row in query_data.values]):
                raise IncorrectQueryException()

            self.insert(query_data)

        elif query_upper.startswith("CREATE TABLE"):
            pass
        elif query_upper.startswith("SELECT"):
            pass
        # TODO: Table manager functions
        else:
            raise QueryNotSupportedException()

    def get_all_tables(self) -> List[str]:
        tables = self.columns.keys()
        return [
            table[0] for table in tables if table[0] not in self.internal_tables.keys()
        ]

    def get_all_columns(self, table_name: str) -> List[Tuple[str, str]]:
        return self.columns[table_name]

    def get_database_state(self) -> Dict[str, List[str]]:
        return {
            table: [column[0] for column in columns]
            for table, columns in self.columns.items()
        }

    def create_table(
        self, table_name: str, column_names: List[str], column_types: List[str]
    ) -> None:
        self.columns[table_name] = list(zip(column_names, column_types))
        self.values[table_name] = []

    def create_internal_tables(self) -> None:
        for table, columns in self.internal_tables.items():
            self.create_table(table, columns[0], columns[1])

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
        for table in self.get_all_tables():
            self.remove_table(table)

        for table in self.internal_tables.keys():
            self.values[table] = []

    def insert(self, query_data: QueryData) -> None:
        """Inserts the data into the database"""
        for query_row in query_data.values:
            database_row = [None for column in self.columns[query_data.table]]

            for query_column, value in zip(query_data.columns, query_row):
                column_index = [
                    i
                    for i, column in enumerate(self.columns[query_data.table])
                    if column[0] == query_column
                ]
                if len(column_index) != 1:
                    raise IncorrectQueryException()

                database_row[column_index[0]] = value

            self.values[query_data.table].append(database_row)
