import logging
import re
from copy import deepcopy
from typing import Any, Dict, List

from ...data.query_data import QueryData
from ..strategy import Strategy
from .openai import _openai_execute_request


class OpenAIStrategy(Strategy):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo-0125",
        max_column_mapping_retries: int = 3,
    ) -> None:
        self.model = model
        self.logger = logging.getLogger(__name__)

        self.max_tokens = 30
        self.max_column_mapping_retries = max_column_mapping_retries

    def generate_table_prediction_request(
        self, query_data: QueryData
    ) -> Dict[str, Any]:
        database_string = (
            "\n".join(
                [
                    f"Table {table}:\n"
                    f"{';'.join([column for column in table_data[0]])}\n"
                    "\n".join([";".join([str(value) for value in row]) for row in table_data[1]])
                    for table, table_data in query_data.database_state.items()
                ]
            )
            if len(query_data.database_state) > 0
            else "No table exists yet."
        )

        return {
            "model": self.model,
            "messages": [
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
            ],
            "max_tokens": self.max_tokens,
        }

    def predict_table_name(self, query_data: QueryData) -> str:
        return (
            re.search(
                r"Table: (?P<table>\S+)",
                (
                    _openai_execute_request(
                        self.generate_table_prediction_request(query_data)
                    )[0]
                    .choices[0]
                    .message.content
                ),
            )
            .group("table")
            .strip()
        )

    def generate_column_mapping_request(
        self,
        query_data: QueryData,
        query_value: str,
        query_column: str,
        db_columns: List[str],
        db_values: List[str],
    ) -> Dict[str, Any]:
        table_string = f"Table {query_data.table}:\n" + "\n".join(
            [
                f"Column {db_column}, Example values: [{', '.join([str(row[db_column_index]) for row in db_values if row[db_column_index] is not None])}]"
                for db_column_index, db_column in enumerate(db_columns)
            ]
        )

        return {
            "model": self.model,
            "messages": [
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
            "max_tokens": self.max_tokens,
        }

    def predict_column_mapping(self, query_data: QueryData) -> List[str]:
        predicted_columns = []

        if query_data.table in query_data.database_state.keys():
            db_columns, db_values = deepcopy(
                query_data.database_state[query_data.table]
            )
        else:
            db_columns, db_values = [], []

        for index, query_value in enumerate(query_data.values[0]):
            query_column = (
                query_data.columns[index]
                if query_data.columns
                else "No column specified"
            )

            request = self.generate_column_mapping_request(
                query_data, query_value, query_column, db_columns, db_values
            )

            # Different columns from the query must not be mapped to the same database column
            # If that is the case retry as long as specified
            # If not one generated name fits, the last prediction is used and an integer is added to its end
            was_added = False
            for i in range(self.max_column_mapping_retries):
                prediction = (
                    re.search(
                        r"Column: (?P<column>\S+)",
                        (
                            _openai_execute_request(request)[0]
                            .choices[0]
                            .message.content
                        ),
                    )
                    .group("column")
                    .strip()
                )

                if prediction not in predicted_columns:
                    predicted_columns.append(prediction)
                    was_added = True
                    break

            # Modify prediction and add if no fitting name was found
            if not was_added:
                modification = 1
                while (
                    modified_prediction := prediction + modification
                ) in predicted_columns:
                    modification += 1
                predicted_columns.append(modified_prediction)
                self.logger.info(
                    f"No fitting db column found for column: {query_column}, value: {query_value} in query: {query_data.get_query()}."
                    f"Used column {modified_prediction} instead."
                )

            # Changes table state, so that predicted column is no longer included
            if prediction in db_columns:
                column_index = db_columns.index(prediction)
                db_columns.remove(prediction)
                db_values = [
                    [value for i, value in enumerate(row) if i != column_index]
                    for row in db_values
                ]

        return predicted_columns
