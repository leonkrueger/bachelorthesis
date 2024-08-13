from typing import Any, List, Tuple

from .data.query_data import QueryData
from .data.table_origin import TableOrigin
from .databases.database import Database
from .insert_query_parser import parse_insert_query
from .strategies.strategy import Strategy
from .table_manager import TableManager


class InsertQueryHandler:
    def __init__(
        self, database: Database, table_manager: TableManager, strategy: Strategy
    ) -> None:
        self.database = database
        self.table_manager = table_manager
        self.strategy = strategy

    def handle_insert_query(self, query: str) -> List[Tuple[Any, ...]]:
        """Collects all required information for the execution of the insert query and executes it on the database"""

        # Parse the query of the user and collect information needed for execution
        query_data = QueryData(query, self.database.get_database_state())
        parse_insert_query(query_data)
        query_data.table_origin = TableOrigin.PREDICTION  # TODO

        query_data.table = self.strategy.predict_table_name(query_data)

        if query_data.table not in query_data.database_state.keys():
            # Create database table if it does not already exist
            self.table_manager.create_table(query_data)
        else:
            # Check if the table name needs to be updated
            # This could be the case, if it was specified by the user
            # If it was not specified or not changed, the name in the database needs to be used
            # if "table" in query_data.keys():
            #     if not self.table_manager.check_update_of_table_name(
            #         table_name, query_data.table
            #     ):
            #         query_data.table = table_name
            # else:
            #     query_data.table = table_name

            # Create not-existing columns
            for index, column in enumerate(query_data.columns):
                if not column in query_data.database_state[query_data.table][0]:
                    self.table_manager.create_column(
                        query_data.table,
                        column,
                        query_data.column_types[index],
                    )

        # Execute constructed query
        return self.database.execute_query(query_data.get_query())
