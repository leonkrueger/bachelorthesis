from typing import Any


class Database:
    def execute(
        self, statement: str, params: tuple[str, ...] = ()
    ) -> list[tuple[Any, ...]]:
        """Executes the statement with the given parameters on the database and returns its result"""
        raise NotImplementedError()

    def select_all_data(self, table_name: str) -> list[tuple[Any, ...]]:
        """Returns the data from the specified table"""
        raise NotImplementedError()

    def get_all_tables(self) -> list[str]:
        """Returns all table names of the database"""
        raise NotImplementedError()

    def get_all_columns(self, table_name: str) -> list[tuple[str, str]]:
        """Returns all column names and types of the given table"""
        raise NotImplementedError()

    def get_example_rows(self, table_name: str) -> list[tuple[Any, ...]]:
        """Returns example rows of the given table"""
        raise NotImplementedError()

    def get_database_state(self) -> dict[str, tuple[list[str], list[list[Any]]]]:
        """Returns a dictionary containing all tables, its columns and example rows"""
        raise NotImplementedError()

    def create_table(
        self, table_name: str, column_names: list[str], column_types: list[str]
    ) -> None:
        """Creates the specified table"""
        raise NotImplementedError()

    def create_column(
        self, table_name: str, column_name: str, column_type: str
    ) -> None:
        """Creates the specified column"""
        raise NotImplementedError()

    def remove_table(self, table_name: str) -> None:
        """Removes the specified table"""
        raise NotImplementedError()

    def reset_database(self) -> None:
        """Removes all custom tables in the database and empties the internal ones"""
        raise NotImplementedError()

    def close(self) -> None:
        """Closes database connections if necessary"""
        pass
