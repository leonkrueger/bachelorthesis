import re
from typing import Dict, List

from ...data.query_data import QueryData
from ..strategy import Strategy
from .openai import (
    _openai_compute_approximate_max_cost,
    _openai_execute_request,
    openai_execute,
)


class OpenAIModel(Strategy):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo-0125",
    ) -> None:
        self.model = model
        self.max_tokens = 30

    def generate_request(self, query_data: QueryData) -> List[Dict[str]]:
        database_string = (
            "\n".join(
                [
                    f"- Table: {table}, Columns: [{', '.join([column[0] for column in columns])}]"
                    for table, columns in query_data.database_state.items()
                ]
            )
            if len(query_data.database_state) > 0
            else "No table exists yet"
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
        return re.search(
            r"Table: (?P<table>\S+)",
            (
                _openai_execute_request(self.generate_request(query_data))[0]
                .choices[0]
                .message.content
            ),
        ).group("table")

    def compute_approximate_max_cost(self, query_data: QueryData) -> float:
        return _openai_compute_approximate_max_cost(self.generate_request(query_data))
