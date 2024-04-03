from .data.query_data import QueryData
from .insert_query_parser import parse_insert_query
from .sql_handler import SQLHandler
from .strategies.strategy import Strategy
from .table_manager import TableManager


def handle_insert_query(
    query: str,
    sql_handler: SQLHandler,
    table_manager: TableManager,
    strategy: Strategy,
) -> list[tuple[str | int | float, ...]]:
    # Parse the query of the user and collect information needed for execution
    query_data = QueryData(query, sql_handler.get_database_state())
    parse_insert_query(query_data)

    query_data.table = strategy.predict_table_name(query_data)

    if query_data.table not in query_data.database_state.keys():
        # Create database table if it does not already exist
        table_manager.create_table(query_data)
    else:
        # Check if the table name needs to be updated
        # This could be the case, if it was specified by the user
        # If it was not specified or not changed, the name in the database needs to be used
        # if "table" in query_data.keys():
        #     if not table_manager.check_update_of_table_name(
        #         table_name, query_data.table
        #     ):
        #         query_data.table = table_name
        # else:
        #     query_data.table = table_name

        # Create not-existing columns
        for index, column in enumerate(query_data.columns):
            if not column in query_data.database_state[query_data.table]:
                table_manager.create_column(
                    query_data.table,
                    column,
                    query_data.column_types[index],
                )

    # Create SQL-query that is run on the database
    row_values_strings = [
        f"({', '.join(row_values)})" for row_values in query_data.values
    ]
    constructed_query = f"INSERT INTO {query_data.table} ({', '.join(query_data.columns)}) VALUES {', '.join(row_values_strings)};"

    # Execute constructed query
    return sql_handler.execute_query(constructed_query)[0]
