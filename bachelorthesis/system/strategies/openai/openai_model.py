import re

from openai import OpenAI

from ...data.query_data import QueryData
from ..strategy import Strategy


class OpenAIModel(Strategy):
    def __init__(
        self,
        openai_api_key: str,
        openai_org_id: str,
        model: str = "gpt-3.5-turbo",
    ) -> None:
        self.client = OpenAI(openai_api_key, openai_org_id)
        self.model = model

    def run_prompt(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        """Runs a prompt an OpenAI chat model and returns its answer"""
        return (
            self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
            )
            .choices[0]
            .message.content
        )

    def predict_table_name(self, query_data: QueryData) -> str:
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

        return re.search(
            r"Table: (?P<table>\S+)",
            self.run_prompt(
                [
                    {
                        "role": "system",
                        "content": "You are an intelligent database that predicts on which table a SQL-insert should be executed. "
                        "The inserts can contain abbreviated or synonymous names. The table and column names can be missing entirely. "
                        "You should then predict your result based on the available information. "
                        "You give the output in the form 'Table: {table_name}'. If there is a suitable table in the database, "
                        "you replace '{table_name}' with its name. Else, you replace '{table_name}' with a suitable name for a database table. "
                        "You don't give any explanation for your result.",
                    },
                    {
                        "role": "user",
                        "content": f"Predict the table for this example:\n"
                        f"Query: {query_data.query}\n"
                        f"Database State:\n{database_string}",
                    },
                ],
                5,
            ),
        ).group("table")
