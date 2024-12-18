from typing import Any

from ..create_table_parser import parse_create_table
from ..data.insert_data import InsertData
from ..insert_parser import parse_insert
from .database import Database


class StatementNotSupportedException(Exception):
    def __init__(self, statement: str):
        self.statement = statement


class IncorrectInsertException(Exception):
    def __init__(self, insert: str):
        self.insert = insert


class insertDoesNotFitDatabaseException(Exception):
    def __init__(self, insert: str, database_columns: list[str]):
        self.insert = insert
        self.database_columns = database_columns


class PythonDatabase(Database):
    columns: dict[str, list[tuple[str, str]]] = {}
    values: dict[str, list[list[Any]]] = {}

    def execute(
        self, statement: str, params: tuple[str, ...] = ()
    ) -> list[tuple[Any, ...]]:
        statement_upper = statement.upper()
        if statement_upper.startswith("INSERT INTO"):
            insert_data = InsertData(statement, {})
            parse_insert(insert_data)

            if not insert_data.table:
                raise IncorrectInsertException(statement)

            if not insert_data.columns and len(insert_data.values[0]) != len(
                self.columns[insert_data.table]
            ):
                raise IncorrectInsertException(statement)
            elif not insert_data.columns:
                insert_data.columns = [
                    column[0] for column in self.get_all_columns(insert_data.table)
                ]

            if any(
                [len(row) != len(insert_data.columns) for row in insert_data.values]
            ):
                raise IncorrectInsertException(statement)

            self.insert(insert_data)

        elif statement_upper.startswith("CREATE TABLE"):
            table_name, column_names, column_types = parse_create_table(statement)
            self.create_table(table_name, column_names, column_types)
        elif statement_upper.startswith("SHOW TABLES"):
            return [self.get_all_tables()]
        elif statement_upper.startswith("SHOW COLUMNS FROM"):
            table_name = statement[18:-1] if statement.endswith(";") else statement[18:]
            return self.get_all_columns(table_name)
        elif statement_upper.startswith("SELECT * FROM"):
            table_name = statement[14:-1] if statement.endswith(";") else statement[14:]
            return self.select_all_data(table_name)
        else:
            raise StatementNotSupportedException(statement)

    def select_all_data(self, table_name: str) -> list[tuple[Any, ...]]:
        return [tuple(row) for row in self.values[table_name]]

    def get_all_tables(self) -> list[str]:
        return self.columns.keys()

    def get_all_columns(self, table_name: str) -> list[tuple[str, str]]:
        return self.columns[table_name]

    def get_example_rows(self, table_name: str) -> list[tuple[Any, ...]]:
        return [tuple(row) for row in self.values[table_name][:3]]

    def get_database_state(self) -> dict[str, tuple[list[str], list[tuple[Any, ...]]]]:
        return {
            table: (
                [column[0] for column in self.get_all_columns(table)],
                self.get_example_rows(table),
            )
            for table in self.get_all_tables()
        }

    def create_table(
        self, table_name: str, column_names: list[str], column_types: list[str]
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

    def insert(self, insert_data: InsertData) -> None:
        """Inserts the data into the database"""
        for insert_row in insert_data.values:
            database_row = [None for column in self.columns[insert_data.table]]

            for insert_column, insert_colummn_type, value in zip(
                insert_data.columns, insert_data.column_types, insert_row
            ):
                column_index = [
                    i
                    for i, column in enumerate(self.columns[insert_data.table])
                    if column[0] == insert_column
                ]
                if len(column_index) != 1:
                    raise insertDoesNotFitDatabaseException(
                        insert_data.insert, self.columns[insert_data.table]
                    )

                database_row[column_index[0]] = (
                    None
                    if value == "NULL"
                    else (
                        int(value)
                        if insert_colummn_type == "BIGINT"
                        else (
                            float(value)
                            if insert_colummn_type == "DOUBLE"
                            else value[1 : len(value) - 1]
                        )
                    )
                )

            self.values[insert_data.table].append(database_row)
