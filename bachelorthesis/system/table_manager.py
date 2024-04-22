from typing import Any

from .data.query_data import QueryData
from .data.table_origin import TableOrigin
from .databases.database import Database


class TableManager:
    def __init__(self, database: Database) -> None:
        self.database = database
        self.database.create_internal_tables()

    def create_table(self, query_data: QueryData) -> None:
        """Creates the specified table and adds it to the table registry"""
        self.database.execute_query(
            f"INSERT INTO table_registry VALUES ('{query_data.table}', '{query_data.table_origin.value}');"
        )

        self.database.create_table(
            query_data.table,
            query_data.columns,
            query_data.column_types,
        )

    def check_update_of_table_name(self, current_table_name: str, new_table_name: str):
        """Checks if the name of the table was set by prediction. If so, it changes its name to the specified one."""
        current_origin = TableOrigin(
            self.database.execute_query(
                f"SELECT name_origin FROM table_registry WHERE name = '{current_table_name}';"
            )[0][0][0]
        )

        if current_origin == TableOrigin.PREDICTION:
            self.database.execute_query(
                f"ALTER TABLE {current_table_name} RENAME {new_table_name};"
            )
            self.database.execute_query(
                f"UPDATE table_registry SET name = '{new_table_name}', name_origin = '{TableOrigin.USER.value}' "
                f"WHERE name = '{current_table_name}';"
            )
            return True

        return False

    def create_column(
        self,
        table_name: str,
        column_name: str,
        column_type: str,
    ) -> None:
        """Creates the specified column"""
        self.database.create_column(table_name, column_name, column_type)
