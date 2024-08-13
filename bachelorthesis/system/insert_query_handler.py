from typing import Any, List, Tuple

from .data.query_data import QueryData
from .databases.database import Database
from .insert_query_parser import parse_insert_query
from .strategies.strategy import Strategy


class InsertQueryHandler:
    def __init__(self, database: Database, strategy: Strategy) -> None:
        self.database = database
        self.strategy = strategy

    def handle_insert_query(self, query: str) -> List[Tuple[Any, ...]]:
        """Collects all required information for the execution of the insert query and executes it on the database"""

        # Parse the query of the user and collect information needed for execution
        query_data = QueryData(query, self.database.get_database_state())
        parse_insert_query(query_data)

        query_data.table = self.strategy.predict_table_name(query_data)
        query_data.columns = self.strategy.predict_column_mapping(query_data)

        if query_data.table not in query_data.database_state.keys():
            # Create database table if it does not already exist
            self.database.create_table(
                query_data.table,
                query_data.columns,
                query_data.column_types,
            )
        else:
            # Create not-existing columns
            for index, column in enumerate(query_data.columns):
                if not column in query_data.database_state[query_data.table][0]:
                    self.database.create_column(
                        query_data.table,
                        column,
                        query_data.column_types[index],
                    )

        # Execute constructed query
        return self.database.execute_query(query_data.get_query())
