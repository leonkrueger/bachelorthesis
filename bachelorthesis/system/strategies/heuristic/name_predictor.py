from ...data.query_data import QueryData
from ...utils.utils import remove_quotes
from ..large_language_model import LargeLanguageModel


class NamePredictor:
    def __init__(self, model: LargeLanguageModel) -> None:
        self.model = model

    def predict_table_name(self, query_data: QueryData) -> str:
        """Predicts a suitable table name for an insertion query"""
        messages = [
            {
                "role": "system",
                "content": "Given a SQL insert query, you should predict a name for a database table that is suitable to the information of the query. "
                "Answer only with the predicted name of the table. Don't give any explanation for your result.",
            },
            {
                "role": "user",
                "content": "Predict a name for a database table for this insert query.\n"
                f"Query: {''.join(query_data.query)}\n"
                "Table:",
            },
        ]

        return remove_quotes(self.model.run_prompt(messages))

    def predict_column_name(self, query_data: QueryData, value: str) -> str:
        """Predicts a suitable column name for a specific value of a query"""
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
                f"Query: {''.join(query_data.query)}\n"
                f"Value: {value}"
                "Column:",
            },
        ]

        return remove_quotes(self.model.run_prompt(messages))
