from typing import Any
from utils.io.console_handler import yes_or_no_question
from utils.models.name_predictor import NamePredictor

from utils.io.sql_handler import SQLHandler
from utils.enums.table_origin import TableOrigin


MINIMAL_COLUMNS_FOUND_RATIO = 0.4


def map_table_to_database(
    query_data: dict[str, Any], sql_handler: SQLHandler, name_predictor: NamePredictor
) -> tuple[str, dict[str, str], bool]:
    # Find the database table that fits the specified arguments in the query best.
    # If there is no table found, the table specified in the query is returned.
    # The second returned value is a mapping of the columns in the query to the columns in the table.
    # The third returned value signals, if the table needs to be created.
    database_tables = sql_handler.get_all_tables()
    if "table" in query_data.keys():
        # Table name specified in the query
        fitting_db_tables = get_fitting_tables(database_tables, query_data["table"])

        if len(fitting_db_tables) == 0:
            # There is no table that fits the specified name
            table_data = map_table_to_database_on_columns(
                query_data, sql_handler, database_tables
            )

            # If columns fit to a table, ask user for feedback
            if table_data[
                "columns_found_ratio"
            ] >= MINIMAL_COLUMNS_FOUND_RATIO and yes_or_no_question(
                f"Found table {table_data['table']} with fitting columns {', '.join(table_data['table_columns'])}. "
                "Do you want use this table for insertion?",
                True,
            ):
                # Use the found table despite having a non-fitting name
                return (table_data["table"], table_data["column_mapping"], False)
            else:
                # Create a new table
                query_data["table_origin"] = TableOrigin.USER
                return (query_data["table"], {}, True)
        else:
            # There is at least one table that fit the specified name
            table_data = map_table_to_database_on_columns(
                query_data, sql_handler, fitting_db_tables
            )

            # If columns do not fit to the table, ask user for feedback
            if table_data[
                "columns_found_ratio"
            ] >= MINIMAL_COLUMNS_FOUND_RATIO or yes_or_no_question(
                f"Found table {table_data['table']} with columns {', '.join(table_data['table_columns'])}. "
                "Do you want use this table for insertion?",
                True,
            ):
                # Use the found table if columns fit as well or if user wants to use this table
                return (
                    table_data["table"],
                    table_data["column_mapping"],
                    False,
                )
            else:
                # Create a new table
                query_data["table_origin"] = TableOrigin.USER
                return (query_data["table"], {}, True)

    else:
        # No table name specified in the query
        table_data = map_table_to_database_on_columns(
            query_data, sql_handler, database_tables
        )

        if table_data["columns_found_ratio"] >= MINIMAL_COLUMNS_FOUND_RATIO:
            # Columns fit to a table in the database
            return (
                table_data["table"],
                table_data["column_mapping"],
                False,
            )
        else:
            # Columns do not fit a table in the database
            predicted_table_name = name_predictor.predict_table_name(
                query_data["columns"]
            )
            query_data["table_origin"] = TableOrigin.PREDICTION
            return (predicted_table_name, {}, True)


def get_fitting_tables(db_tables: list[str], table: str) -> list[str]:
    # Returns all tables whose name fits the specified table.
    return [db_table for db_table in db_tables if table.lower() == db_table.lower()]


def map_table_to_database_on_columns(
    query_data: dict[str, Any], sql_handler: SQLHandler, tables_to_consider: list[str]
) -> dict[str, Any]:
    # Find the one out of the preselected database tables that fits the specified columns best.
    return best_fitting_columns(
        sql_handler,
        tables_to_consider,
        query_data["columns"],  # TODO: columns dont need to be defined
        query_data["column_types"],
    )


def best_fitting_columns(
    sql_handler: SQLHandler,
    db_tables: list[str],
    columns: list[str],
    column_types: list[str],
) -> dict[str, Any]:
    # Checks all tables of the database for the one that contains most of the specified columns.
    # Returns a dict with the required information of the best fitting table
    best_table_data = {
        "table": None,
        "table_columns": [],
        "column_mapping": {},
        "columns_found_ratio": 0.0,
    }

    # The first element is the unchanged column name, as MySQL is case sensitive. The second element is used for comparison.
    query_columns = [
        (column, column.lower(), column_type.lower())
        for column, column_type in zip(columns, column_types)
    ]

    # Try to map the columns for every table
    for db_table in db_tables:
        # The first element is the unchanged column name, as MySQL is case sensitive.
        # The second element is used for comparison and the third is the type of the column.
        db_table_columns = [
            (column_info[0], column_info[0].lower(), column_info[1].lower())
            for column_info in sql_handler.get_all_columns(db_table)
        ]
        column_mapping = {}

        columns_found = 0
        # Test for every combination of query columns and table columns, if they mtach
        for query_column in query_columns:
            column_found = False
            for db_table_column in db_table_columns:
                # They match, if both its type and its name are identical
                if (
                    query_column[2] == db_table_column[2]
                    and query_column[1] == db_table_column[1]
                ):
                    columns_found += 1
                    column_mapping[query_column[0]] = db_table_column[0]
                    column_found = True
            if column_found:
                continue

        if columns_found / len(columns) > best_table_data["columns_found_ratio"]:
            best_table_data["columns_found_ratio"] = columns_found / len(columns)
            best_table_data["table"] = db_table
            best_table_data["table_columns"] = [
                column[0] for column in db_table_columns
            ]
            best_table_data["column_mapping"] = column_mapping

    return best_table_data
