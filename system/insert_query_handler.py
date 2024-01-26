from utils.name_predictor import NamePredictor
from system.insert_query_parser import parse_insert_query
from system.table_mapper import map_table_to_database
from utils.sql_handler import SQLHandler
from utils.table_manager import TableManager


def handle_insert_query(
    query: str,
    sql_handler: SQLHandler,
    table_manager: TableManager,
    name_predictor: NamePredictor,
) -> list[tuple[str | int | float, ...]]:
    # Parse the query of the user and collect information needed for execution
    query_data = parse_insert_query(query)
    query_data["table"], column_mapping, create_table = map_table_to_database(
        query_data, sql_handler, name_predictor
    )

    if create_table:
        # Create database table if it does not already exist
        table_manager.create_table(query_data)
    else:
        # Select correct column names and create not-existing columns
        db_columns = []
        for column_index, column in enumerate(query_data["columns"]):
            if column in column_mapping.keys():
                db_columns.append(column_mapping[column])
            else:
                table_manager.create_column(
                    query_data["table"],
                    column,
                    [row[column_index] for row in query_data["values"]],
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
