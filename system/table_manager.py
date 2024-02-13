from typing import Any
from utils.io.sql_handler import SQLHandler


class TableManager:
    def __init__(self, sql_handler: SQLHandler) -> None:
        self.sql_handler = sql_handler
        self.sql_handler.create_internal_tables()

    def create_table(self, query_data: dict[str, Any]) -> None:
        # Creates the specified table and adds it to the table registry
        self.sql_handler.execute_query(
            f"""INSERT INTO table_registry VALUES ('{query_data['table']}', '{query_data['table_origin'].value}');"""
        )

        self.sql_handler.create_table(
            query_data["table"],
            query_data["columns"],
            query_data["column_types"],
        )

    def create_column(
        self,
        table_name: str,
        column_name: str,
        column_type: str,
    ) -> None:
        # Creates the specified column
        self.sql_handler.create_column(table_name, column_name, column_type)
