from typing import Any, Dict, List, Tuple

from ...data.query_data import QueryData
from ...data.table_origin import TableOrigin
from ..strategy import Strategy
from .name_predictor import NamePredictor


class HeuristicStrategy(Strategy):
    MINIMAL_COLUMNS_FOUND_RATIO_RELAXED = 0.4
    MINIMAL_COLUMNS_FOUND_RATIO_STRICT = 0.7

    # Is filled when the table name is predicted and resetted after the last prediction step for a query
    # First value is the name of the table, second the column mapping
    predictions = None

    def __init__(self, name_predictor: NamePredictor) -> None:
        super().__init__()

        self.name_predictor = name_predictor

    def predict_table_name(self, query_data: QueryData) -> str:
        if self.predictions is None:
            self.predictions = self.map_table_to_database(query_data)

        # TODO: Remove following two lines and change third
        table_name = self.predictions[0]
        self.predictions = None
        return table_name

    def predict_column_mapping(self, query_data: QueryData) -> Dict[str, str]:
        column_mapping = self.predictions[1]
        self.predictions = None
        return column_mapping

    def map_table_to_database(
        self,
        query_data: QueryData,
    ) -> Tuple[str, Dict[str, str], bool]:
        """Find the database table that fits the specified arguments in the query best.
        If there is no table found, the table specified in the query is returned.
        The second returned value is a mapping of the columns in the query to the columns in the table.
        The third returned value signals, if the table needs to be created."""
        if query_data.table is not None:
            # Table name specified in the query
            fitting_db_tables = self.get_fitting_tables(
                query_data.database_state.keys(), query_data.table
            )

            if len(fitting_db_tables) == 0:
                # There is no table that fits the specified name
                table_data = self.map_table_to_database_on_columns(
                    query_data, query_data.database_state
                )

                if (
                    table_data["columns_found_ratio"]
                    >= self.MINIMAL_COLUMNS_FOUND_RATIO_STRICT
                ):
                    # Use the found table despite having a non-fitting name if the columns fit really well
                    return (table_data["table"], table_data["column_mapping"])
                else:
                    # Create a new table else
                    query_data.table_origin = TableOrigin.USER
                    return (query_data.table, {})
            else:
                # There is at least one table that fit the specified name
                table_data = self.map_table_to_database_on_columns(
                    query_data,
                    {
                        table: columns
                        for table, columns in query_data.database_state.items()
                        if table in fitting_db_tables
                    },
                )

                # Use the table, with the best fitting columns
                return (table_data["table"], table_data["column_mapping"])

                # if table_data[
                #     "columns_found_ratio"
                # ] >= self.MINIMAL_COLUMNS_FOUND_RATIO_RELAXED:
                #     # Use the found table if columns fit as well
                #     return (
                #         table_data["table"],
                #         table_data["column_mapping"],
                #         False,
                #     )
                # else:
                #     # Create a new table
                #     query_data.table_origin = TableOrigin.USER
                #     return (query_data.table, {}, True)

        else:
            # No table name specified in the query
            table_data = self.map_table_to_database_on_columns(
                query_data, query_data.database_state
            )

            if (
                table_data["columns_found_ratio"]
                >= self.MINIMAL_COLUMNS_FOUND_RATIO_RELAXED
            ):
                # Columns fit to a table in the database
                return (table_data["table"], table_data["column_mapping"])
            else:
                # Columns do not fit a table in the database
                predicted_table_name = self.name_predictor.predict_table_name(
                    query_data.columns
                )
                query_data.table_origin = TableOrigin.PREDICTION
                return (predicted_table_name, {})

    def get_fitting_tables(self, db_tables: List[str], table: str) -> List[str]:
        """Returns all tables whose name fits the specified table."""
        return [db_table for db_table in db_tables if table == db_table]

    def map_table_to_database_on_columns(
        self,
        query_data: QueryData,
        tables_to_consider: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Find the one out of the preselected database tables that fits the specified columns best."""
        return self.best_fitting_columns(
            tables_to_consider,
            query_data.columns,  # TODO: columns dont need to be defined
            # query_data.column_types,
        )

    def best_fitting_columns(
        self,
        db_tables: Dict[str, List[str]],
        query_columns: List[str],
        # column_types: List[str],
    ) -> Dict[str, Any]:
        """Checks all tables of the database for the one that contains most of the specified columns.
        Returns a dict with the required information of the best fitting table"""
        best_table_data = {
            "table": None,
            "table_columns": [],
            "column_mapping": {},
            "columns_found_ratio": 0.0,
        }

        # # The first element is the unchanged column name, as MySQL is case sensitive. The second element is used for comparison.
        # query_columns = [
        #     (column, column.lower(), column_type.lower())
        #     for column, column_type in zip(columns, column_types)
        # ]

        # Try to map the columns for every table
        for db_table, db_columns in db_tables.items():
            # # The first element is the unchanged column name, as MySQL is case sensitive.
            # # The second element is used for comparison and the third is the type of the column.
            # db_table_columns = [
            #     (column_info[0], column_info[0].lower(), column_info[1].lower())
            #     for column_info in sql_handler.get_all_columns(db_table)
            # ]
            column_mapping = {}

            columns_found = 0
            # Test for every combination of query columns and table columns, if they mtach
            for query_column in query_columns:
                column_found = False
                for db_column in db_columns:
                    # They match, if both its type and its name are identical
                    # if (
                    #     query_column[2] == db_table_column[2]
                    #     and query_column[1] == db_table_column[1]
                    # ):
                    if query_column == db_column:
                        columns_found += 1
                        column_mapping[query_column[0]] = db_column[0]
                        column_found = True
                if column_found:
                    continue

            if (
                columns_found / len(query_columns)
                > best_table_data["columns_found_ratio"]
            ):
                best_table_data["columns_found_ratio"] = columns_found / len(
                    query_columns
                )
                best_table_data["table"] = db_table
                best_table_data["table_columns"] = query_columns
                best_table_data["column_mapping"] = column_mapping

        return best_table_data
