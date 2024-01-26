import random
from typing import Any
from utils.name_predictor import NamePredictor

from utils.sql_handler import SQLHandler
from utils.table_origin import TableOrigin


MINIMAL_COLUMNS_FOUND_RATIO = 0.4


def map_table_to_database(
    query_data: dict[str, Any], sql_handler: SQLHandler, name_predictor: NamePredictor
) -> tuple[str, dict[str, str], bool]:
    # Find the database table that fits the specified arguments in the query best.
    # If there is no table found, the table specified in the query is returned.
    # The second returned value is a mapping of the columns in the query to the columns in the table.
    # The third returned value signals, if the table needs to be created.
    if "table" in query_data.keys():
        fitting_db_tables = get_fitting_tables(
            sql_handler.get_all_tables(), query_data["table"]
        )

        if len(fitting_db_tables) == 0:
            # There is no table that fits the specified name. Thus a new table is created.
            query_data["table_origin"] = TableOrigin.USER
            return (query_data["table"], {}, True)
        else:
            # There is at least one table that fit the specified name.
            table, column_mapping = map_table_to_database_on_columns(
                query_data, sql_handler, fitting_db_tables
            )
            return (
                table,
                column_mapping,
                False,
            )
    else:
        table, column_mapping, columns_found_ratio = map_table_to_database_on_columns(
            query_data, sql_handler, sql_handler.get_all_tables()
        )

        if columns_found_ratio >= MINIMAL_COLUMNS_FOUND_RATIO:
            return (
                table,
                column_mapping,
                False,
            )
        else:
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
) -> tuple[str, dict[str, str], float]:
    # Find the one out of the preselected database tables that fits the specified columns best.
    # The second returned value is a mapping of the columns in the query to the columns in the table.
    # The third value is the ratio of found columns.
    table, column_mapping, columns_found_ratio = best_fitting_columns(
        sql_handler,
        tables_to_consider,
        query_data["columns"],  # TODO: columns dont need to be defined
    )

    if table:
        # If at least one table contains any specified column,
        # the table that contains the most specified columns is selected.
        return (table, column_mapping, columns_found_ratio)
    else:
        # If no table cotains any of the specified columns, a random table is selected.
        return (random.choice(tables_to_consider), {}, 0.0)


def best_fitting_columns(
    sql_handler: SQLHandler, db_tables: list[str], columns: list[str]
) -> tuple[str | None, dict[str, str], float]:
    # Checks all tables of the database for the one that contains most of the specified columns.
    # Returns the name of the table, the mapping of columns and the ratio of found columns.
    best_column_found_ratio = 0.0
    best_table_found = None
    best_column_mapping = {}

    # The first element is the unchanged column name, as MySQL is case sensitive. The second element is used for comparison.
    query_columns = [(column, column.lower()) for column in columns]

    # Try to map the columns for every table
    for db_table in db_tables:
        # The first element is the unchanged column name, as MySQL is case sensitive. The second element is used for comparison.
        db_table_columns = [
            (column, column.lower()) for column in sql_handler.get_all_columns(db_table)
        ]
        column_mapping = {}

        columns_found = 0
        # Test for every combination of query columns and table columns, if they fit together
        for query_column in query_columns:
            column_found = False
            for db_table_column in db_table_columns:
                if query_column[1] == db_table_column[1]:
                    columns_found += 1
                    column_mapping[query_column[0]] = db_table_column[0]
                    column_found = True
            if column_found:
                continue

        if columns_found / len(columns) > best_column_found_ratio:
            best_column_found_ratio = columns_found / len(columns)
            best_table_found = db_table
            best_column_mapping = column_mapping

    return (best_table_found, best_column_mapping, best_column_found_ratio)
