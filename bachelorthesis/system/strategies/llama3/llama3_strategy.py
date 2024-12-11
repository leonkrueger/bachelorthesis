import logging
import re
from copy import deepcopy
from typing import Dict, List

from ...data.insert_data import InsertData
from ..strategy import Strategy
from .llama3_model import Llama3Model


class Llama3Strategy(Strategy):
    def __init__(
        self,
        table_prediction_model_dir: str = None,
        column_mapping_model_dir: str = None,
        max_column_mapping_retries: int = 5,
        use_model_explanations: bool = False,
    ) -> None:
        self.table_prediction_model = Llama3Model(table_prediction_model_dir)
        self.column_mapping_model = (
            self.table_prediction_model
            if table_prediction_model_dir == column_mapping_model_dir
            else Llama3Model(column_mapping_model_dir)
        )
        self.logger = logging.getLogger(__name__)

        self.max_column_mapping_retries = max_column_mapping_retries
        self.use_model_explanations = use_model_explanations

    def extract_prediction_from_explanation(self, output: str) -> str:
        if last_match := re.findall(
            r"(?:\"|`|\'|\*\*)([A-Za-z0-9_$]+)(?:\"|`|\'|\*\*)",
            output,
        ):
            return last_match[-1]
        if last_match := re.findall(r":\s*([A-Za-z0-9_$]+)", output):
            return last_match[-1]
        if last_match := re.findall(r"([A-Za-z0-9_$]+)", output):
            return last_match[-1]
        return "incorrect_table_prediction"

    def predict_table_name(self, insert_data: InsertData) -> str:
        database_string = (
            "\n".join(
                [
                    f"Table {table}:\n"
                    f"{';'.join([column for column in table_data[0]])}\n"
                    "\n".join(
                        [
                            ";".join([str(value) for value in row])
                            for row in table_data[1]
                        ]
                    )
                    for table, table_data in insert_data.database_state.items()
                ]
            )
            if len(insert_data.database_state) > 0
            else "No table exists yet."
        )

        output = self.table_prediction_model.run_prompt(
            [
                {
                    "role": "system",
                    "content": "You are an intelligent database that predicts on which table a SQL-insert should be executed. "
                    "The inserts can contain abbreviated or synonymous names. The table and column names can be missing entirely. "
                    "Base your guess on the available information. "
                    "If there is a suitable table in the database answer its name. Else, predict a suitable name for a new database table. "
                    + (
                        "Explain your answer. Include your final answer in the last sentence."
                        if self.use_model_explanations
                        else "Answer only with the name of the table. Don't give any explanation for your result."
                    ),
                },
                {
                    "role": "user",
                    "content": "Predict the table for this example:\n"
                    f"Query: {insert_data.get_insert(use_quotes=False)}\n"
                    f"Database State:\n{database_string}\n"
                    + ("" if self.use_model_explanations else "Table:"),
                },
            ]
        )

        if self.use_model_explanations:
            return self.extract_prediction_from_explanation(output)
        else:
            return re.search(r"(?P<table>\S+)", output).group("table")

    def predict_column_mapping(self, insert_data: InsertData) -> List[str]:
        predicted_columns = []

        if insert_data.table in insert_data.database_state.keys():
            db_columns, db_values = deepcopy(
                insert_data.database_state[insert_data.table]
            )
        else:
            db_columns, db_values = [], []

        for index, insert_value in enumerate(insert_data.values[0]):
            table_string = f"Table {insert_data.table}:\n" + "\n".join(
                [
                    f"Column {db_column}, Example values: [{', '.join([str(row[db_column_index]) for row in db_values if row[db_column_index] is not None])}]"
                    for db_column_index, db_column in enumerate(db_columns)
                ]
            )

            insert_column = (
                insert_data.columns[index]
                if insert_data.columns
                else "No column specified"
            )

            # Different columns from the insert must not be mapped to the same database column
            # If that is the case retry as long as specified
            # If not one generated name fits, the last prediction is used and an integer is added to its end
            was_added = False
            for i in range(self.max_column_mapping_retries):
                output = self.column_mapping_model.run_prompt(
                    [
                        {
                            "role": "system",
                            "content": "You are an intelligent database that predicts the columns of a SQL-insert. "
                            "The inserts can contain abbreviated or synonymous column names. The column names can also be missing entirely. "
                            "Base your guess on the available information. "
                            "If there is a suitable column in the table answer its name. Else, predict a suitable name for a new column in this table. "
                            + (
                                "Explain your answer. Include your final answer in the last sentence."
                                if self.use_model_explanations
                                else "Answer only with the name of the column. Don't give any explanation for your result."
                            ),
                        },
                        {
                            "role": "user",
                            "content": "Predict the column for this value:\n"
                            f"Query: {insert_data.get_insert(use_quotes=False)}\n"
                            f"Specified column: {insert_column}\n"
                            f"Value: {insert_value}\n"
                            f"{table_string}\n"
                            + ("" if self.use_model_explanations else "Column:"),
                        },
                    ],
                )

                if self.use_model_explanations:
                    prediction = self.extract_prediction_from_explanation(output)
                else:
                    prediction = re.search(r"(?P<column>\S+)", output).group("column")

                if prediction not in predicted_columns:
                    predicted_columns.append(prediction)
                    was_added = True
                    break

            # Modify prediction and add if no fitting name was found
            if not was_added:
                modification = 1
                while (
                    modified_prediction := prediction + str(modification)
                ) in predicted_columns:
                    modification += 1
                predicted_columns.append(modified_prediction)
                self.logger.info(
                    f"No fitting db column found for column: {insert_column}, value: {insert_value} in insert: {insert_data.get_insert()}."
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
