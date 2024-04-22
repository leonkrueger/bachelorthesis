from typing import Any, Dict, List, Tuple


class Database:
    def execute_query(
        self, query: str, params: Tuple[str, ...] = ()
    ) -> List[Tuple[Any, ...]]:
        """Executes the query with the given parameters on the database and returns its result"""
        raise NotImplementedError()

    def get_all_tables(self) -> List[str]:
        """Returns all table names of the database"""
        raise NotImplementedError()

    def get_all_columns(self, table_name: str) -> List[Tuple[str, str]]:
        """Returns all column names and types of the given table"""
        raise NotImplementedError()

    def get_database_state(self) -> Dict[str, List[str]]:
        """Returns a dictionary containing all tables and its columns"""
        raise NotImplementedError()

    def create_table(
        self, table_name: str, column_names: List[str], column_types: List[str]
    ) -> None:
        """Creates the specified table"""
        raise NotImplementedError()

    def create_internal_tables(self) -> None:
        """Creates all tables used for internal database management"""
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

    def close() -> None:
        """Closes database connections if necessary"""
        pass
