import re

from ...data.query_data import QueryData
from ..strategy import Strategy
from .llama3_model import Llama3Model


class Llama3Strategy(Strategy):
    def __init__(
        self,
        model: Llama3Model,
    ) -> None:
        self.model = model

    def predict_table_name(self, query_data: QueryData) -> str:
        database_string = (
            "\n".join(
                [
                    f"- Table: {table}, Columns: [{', '.join([column[0] for column in table_data[0]])}]"
                    for table, table_data in query_data.database_state.items()
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
