from openai import OpenAI

from utils.io.sql_handler import SQLHandler


class OpenAIModel:
    def __init__(self, sql_handler: SQLHandler) -> None:
        self.client = OpenAI()  # Insert keys
        self.sql_handler = sql_handler

    def run_prompt(self, query: str) -> str:
        tables = self.sql_handler.get_all_tables()
        database_state = (
            "\n".join(
                [
                    f"- Table: {table}, Columns: [{', '.join([column[0] for column in self.sql_handler.get_all_columns(table)])}]"
                    for table in tables
                ]
            )
            if len(tables) > 0
            else "No table exists yet."
        )

        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent database that predicts on which table a SQL-insert should be executed. "
                    "You give the output in the form 'Table: {table_name}' where '{table_name}' is replaced with the determined table name "
                    "if there is a suitable table in the database or 'New table' else. You don't give any explanation for your result.",
                },
                {
                    "role": "user",
                    "content": f"Predict the table for this example: \n"
                    f"Query: {query}\n"
                    f"Database State: \n{database_state}",
                },
            ],
            max_tokens=5,
        )

        print(completion)
        return completion.choices[0].message.content
