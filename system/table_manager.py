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
            self.get_column_types(query_data["values"]),
        )

    def create_column(
        self, table_name: str, column_name: str, column_values: list[str]
    ) -> None:
        # Creates the specified column
        self.sql_handler.create_column(
            table_name, column_name, self.get_column_type(column_values[0])
        )

    def get_column_types(self, values: list[list[str]]) -> list[str]:
        # Computes the required column types for the given values
        return [self.get_column_type(value) for value in values[0]]

    def get_column_type(self, value: str) -> str:
        # Computes the required column type for the given value
        if value.startswith('"') or value.startswith("'"):
            return "VARCHAR(255)"
        elif "." in value:
            return "DOUBLE"
        else:
            return "BIGINT"
