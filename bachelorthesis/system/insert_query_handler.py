from bachelorthesis.utils.models.openai_model import OpenAIModel
from system.insert_query_parser import parse_insert_query
from utils.io.sql_handler import SQLHandler
from system.table_manager import TableManager


def handle_insert_query(
    query: str,
    sql_handler: SQLHandler,
    table_manager: TableManager,
    openai_model: OpenAIModel,
) -> list[tuple[str | int | float, ...]]:
    # Parse the query of the user and collect information needed for execution
    query_data = parse_insert_query(query)

    database_state = sql_handler.get_database_state()
    query_data["table"] = openai_model.predict_table_name(query, database_state)

    if query_data["table"] not in database_state.keys():
        # Create database table if it does not already exist
        table_manager.create_table(query_data)
    else:
        # Check if the table name needs to be updated
        # This could be the case, if it was specified by the user
        # If it was not specified or not changed, the name in the database needs to be used
        # if "table" in query_data.keys():
        #     if not table_manager.check_update_of_table_name(
        #         table_name, query_data["table"]
        #     ):
        #         query_data["table"] = table_name
        # else:
        #     query_data["table"] = table_name

        # Create not-existing columns
        for index, column in enumerate(query_data["columns"]):
            if not column in database_state[query_data["table"]]:
                table_manager.create_column(
                    query_data["table"],
                    column,
                    query_data["column_types"][index],
                )

    # Create SQL-query that is run on the database
    row_values_strings = [
        f"({', '.join(row_values)})" for row_values in query_data["values"]
    ]
    constructed_query = f"INSERT INTO {query_data['table']} ({', '.join(query_data['columns'])}) VALUES {', '.join(row_values_strings)};"

    # Execute constructed query
    return sql_handler.execute_query(constructed_query)[0]
