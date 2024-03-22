from utils.models.name_predictor import NamePredictor
from system.insert_query_parser import parse_insert_query
from system.table_mapper import map_table_to_database
from utils.io.sql_handler import SQLHandler
from system.table_manager import TableManager


def handle_insert_query(
    query: str,
    sql_handler: SQLHandler,
    table_manager: TableManager,
    name_predictor: NamePredictor,
) -> list[tuple[str | int | float, ...]]:
    # Parse the query of the user and collect information needed for execution
    query_data = parse_insert_query(query)
    table_name, column_mapping, create_table = map_table_to_database(
        query_data, sql_handler, name_predictor
    )

    if create_table:
        # Create database table if it does not already exist
        query_data["table"] = table_name
        table_manager.create_table(query_data)
    else:
        # Check if the table name needs to be updated
        # This could be the case, if it was specified by the user
        # If it was not specified or not changed, the name in the database needs to be used
        if "table" in query_data.keys():
            if not table_manager.check_update_of_table_name(
                table_name, query_data["table"]
            ):
                query_data["table"] = table_name
        else:
            query_data["table"] = table_name

        # Select correct column names and create not-existing columns
        db_columns = []
        for column, column_type in zip(
            query_data["columns"], query_data["column_types"]
        ):
            if column in column_mapping.keys():
                db_columns.append(column_mapping[column])
            else:
                table_manager.create_column(
                    query_data["table"],
                    column,
                    column_type,
                )
                db_columns.append(column)
        query_data["columns"] = db_columns

    # Create SQL-query that is run on the database
    row_values_strings = [
        f"({', '.join(row_values)})" for row_values in query_data["values"]
    ]
    constructed_query = f"INSERT INTO {query_data['table']} ({', '.join(query_data['columns'])}) VALUES {', '.join(row_values_strings)};"

    # Execute constructed query
    return sql_handler.execute_query(constructed_query)[0]
