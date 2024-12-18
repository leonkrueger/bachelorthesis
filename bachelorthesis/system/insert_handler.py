from typing import Any

from .data.insert_data import InsertData
from .databases.database import Database
from .insert_parser import parse_insert
from .strategies.strategy import Strategy


class InsertHandler:
    def __init__(self, database: Database, strategy: Strategy) -> None:
        self.database = database
        self.strategy = strategy

    def handle_insert(self, insert: str) -> list[tuple[Any, ...]]:
        """Collects all required information for the execution of the insert and executes it on the database"""

        # Parse the insert of the user and collect information needed for execution
        insert_data = InsertData(insert, self.database.get_database_state())
        parse_insert(insert_data)

        insert_data.table = self.strategy.predict_table_name(insert_data)
        insert_data.columns = self.strategy.predict_column_mapping(insert_data)

        if insert_data.table not in insert_data.database_state.keys():
            # Create database table if it does not already exist
            self.database.create_table(
                insert_data.table,
                insert_data.columns,
                insert_data.column_types,
            )
        else:
            # Create not-existing columns
            for index, column in enumerate(insert_data.columns):
                if not column in insert_data.database_state[insert_data.table][0]:
                    self.database.create_column(
                        insert_data.table,
                        column,
                        insert_data.column_types[index],
                    )

        # Execute constructed insert
        return self.database.execute(insert_data.get_insert())
