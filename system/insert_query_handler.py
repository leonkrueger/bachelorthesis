from utils.sql_handler import SQLHandler
from system.insert_query_parser import parse_insert_query
from system.table_mapper import map_table_to_database


def handle_insert_query(
    query: str, sql_handler: SQLHandler
) -> list[tuple[str | int | float, ...]]:
    # Parse the query of the user and collect information needed for execution
    query_data = parse_insert_query(query)
    query_data["table"], column_mapping, create_table = map_table_to_database(
        query_data, sql_handler
    )

    if create_table:
        # Create database table if it does not already exist
        sql_handler.create_table(
            query_data["table"],
            query_data["columns"],
            get_column_types(query_data["values"]),
        )
    else:
        # Select correct column names and create not-existing columns
        db_columns = []
        for column_index, column in enumerate(query_data["columns"]):
            if column in column_mapping.keys():
                db_columns.append(column_mapping[column])
            else:
                sql_handler.create_column(
                    query_data["table"],
                    column,
                    get_column_type(query_data["values"][0][column_index]),
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


def get_column_types(values: list[list[str]]) -> list[str]:
    # Computes the required column types for the given values
    return [get_column_type(value) for value in values[0]]


def get_column_type(value: str) -> str:
    # Computes the required column type for the given value
    if value.startswith('"') or value.startswith("'"):
        return "VARCHAR(255)"
    elif "." in value:
        return "DOUBLE"
    else:
        return "BIGINT"
