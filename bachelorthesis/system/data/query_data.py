from typing import Dict, List, Union

from .table_origin import TableOrigin


class QueryData:
    query: str
    database_state: Dict[str, List[str]]

    table: Union[str, None] = None
    table_origin: Union[TableOrigin, None] = None

    columns: Union[List[str], None] = None
    column_types: Union[List[str], None] = None

    values: Union[List[List[str]], None] = None

    def __init__(self, query: str, database_state: Dict[str, List[str]]):
        self.query = query
        self.database_state = database_state

    def get_query(self, use_quotes: bool = True) -> str:
        """Creates the SQL-query with all available information"""
        table_string = (
            (f"`{self.table}` " if use_quotes else f"{self.table} ")
            if self.table
            else ""
        )
        columns_string = (
            "("
            + ", ".join(
                [f"`{column}`" if use_quotes else column for column in self.columns]
            )
            + ") "
            if self.columns
            else ""
        )
        row_values_strings = [
            f"({', '.join(row_values)})" for row_values in self.values
        ]

        return f"INSERT INTO {table_string}{columns_string}VALUES {', '.join(row_values_strings)};"
