import re
from copy import deepcopy
from typing import Dict, List

from ...data.query_data import QueryData
from ..strategy import Strategy
from .llama3_model import Llama3Model


class Llama3Strategy(Strategy):
    def __init__(
        self,
        table_prediction_model_dir: str = None,
        column_mapping_model_dir: str = None,
    ) -> None:
        self.model = Llama3Model()
        self.table_prediction_model_dir = table_prediction_model_dir
        self.column_mapping_model_dir = column_mapping_model_dir

    def predict_table_name(self, query_data: QueryData) -> str:
        # Loads the correct adapter for the table prediction task
        self.model.load_and_set_adapter(self.table_prediction_model_dir)

        database_string = (
            "\n".join(
                [
                    f"Table {table}:\n"
                    f"{';'.join([column for column in table_data[0]])}\n"
                    "\n".join([";".join(row) for row in table_data[1]])
                    for table, table_data in query_data.database_state
                ]
            )
            if len(query_data.database_state) > 0
            else "No table exists yet."
        )

        return re.search(
            r"(?P<table>\S+)",
            self.model.run_prompt(
                [
                    {
                        "role": "system",
                        "content": "You are an intelligent database that predicts on which table a SQL-insert should be executed. "
                        "The inserts can contain abbreviated or synonymous names. The table and column names can be missing entirely. "
                        "Base your guess on the available information. "
                        "If there is a suitable table in the database answer its name. Else, predict a suitable name for a new database table. "
                        "Answer only with the name of the table. Don't give any explanation for your result.",
                    },
                    {
                        "role": "user",
                        "content": "Predict the table for this example:\n"
                        f"Query: {query_data.get_query(use_quotes=False)}\n"
                        f"Database State:\n{database_string}\n"
                        "Table:",
                    },
                ]
            ),
        ).group("table")

    def predict_column_mapping(self, query_data: QueryData) -> List[str]:
        # Loads the correct adapter for the column mapping task
        self.model.load_and_set_adapter(self.column_mapping_model_dir)

        predicted_columns = []

        if query_data.table in query_data.database_state.keys():
            db_columns, db_values = deepcopy(
                query_data.database_state[query_data.table]
            )
        else:
            db_columns, db_values = [], []

        for index, query_value in enumerate(query_data.values[0]):
            table_string = f"Table {query_data.table}:\n" + "\n".join(
                [
                    f"Column {db_column}, Example values: [{', '.join([row[db_column_index] for row in db_values if row[db_column_index] is not None])}]"
                    for db_column_index, db_column in enumerate(db_columns)
                ]
            )

            query_column = (
                query_data.columns[index]
                if query_data.columns
                else "No column specified"
            )

            predicted = self.model.run_prompt(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are an intelligent database that predicts the columns of a SQL-insert. "
                            "The inserts can contain abbreviated or synonymous column names. The column names can also be missing entirely. "
                            "Base your guess on the available information. "
                            "If there is a suitable column in the table answer its name. Else, predict a suitable name for a new column in this table. "
                            "Answer only with the name of the column. Don't give any explanation for your result."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Predict the column for this value:\n"
                            f"Query: {query_data.get_query(use_quotes=False)}\n"
                            f"Specified column: {query_column}\n"
                            f"Value: {query_value}\n"
                            f"{table_string}\n"
                            "Column:"
                        ),
                    },
                ],
            )
            predicted_columns.append(predicted)

            # Changes table state, so that predicted column is no longer included
            if predicted in db_columns:
                column_index = db_columns.index(predicted)
                db_columns.remove(predicted)
                db_values = [
                    [value for i, value in enumerate(row) if i != column_index]
                    for row in db_values
                ]

        return predicted_columns
