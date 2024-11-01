from ...data.insert_data import InsertData
from ...utils.utils import remove_quotes
from ..large_language_model import LargeLanguageModel


class NamePredictor:
    def __init__(self, model: LargeLanguageModel) -> None:
        self.model = model

    def predict_table_name(self, insert_data: InsertData) -> str:
        """Predicts a suitable table name for an insert"""
        messages = [
            {
                "role": "system",
                "content": "Given a SQL insert query, you should predict a name for a database table that is suitable to the information of the query. "
                "Answer only with the predicted name of the table. Don't give any explanation for your result.",
            },
            {
                "role": "user",
                "content": "Predict a name for a database table for this insert query.\n"
                f"Query: {''.join(insert_data.insert)}\n"
                "Table:",
            },
        ]

        return remove_quotes(self.model.run_prompt(messages)).replace(" ", "_")

    def predict_column_name(self, insert_data: InsertData, value: str) -> str:
        """Predicts a suitable column name for a specific value of an insert"""
        messages = [
            {
                "role": "system",
                "content": "Given a SQL insert query and a specific value, you should predict a name for a database column "
                "that is suitable to store the specified value of the query. "
                "Answer only with the predicted name of the column. Don't give any explanation for your result.",
            },
            {
                "role": "user",
                "content": "Predict a name for a database column for this value.\n"
                f"Query: {''.join(insert_data.insert)}\n"
                f"Value: {value}"
                "Column:",
            },
        ]

        return remove_quotes(self.model.run_prompt(messages)).replace(" ", "_")
