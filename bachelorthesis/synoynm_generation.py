import json
import os
from collections import defaultdict
from pathlib import Path

from system.utils.utils import load_env_variables, remove_quotes

load_env_variables()

from system.strategies.llama3.llama3_model import Llama3Model
from tqdm import tqdm

synonym_generation_data_path = (
    Path(__file__).resolve().parent
    / os.environ["EVALUATION_BASE_DIR"]
    / "fine_tuning"
    / "column_synonym_generation_data.json"
)

synonyms_path = (
    Path(__file__).resolve().parent
    / os.environ["EVALUATION_BASE_DIR"]
    / "fine_tuning"
    / "column_synonyms.json"
)

model = Llama3Model()


def generate_table_synonyms(
    data_point: dict[str, str], synonyms: dict[str, dict[str, list[str]]]
) -> None:
    if len(data_point["query"]) == 0:
        return

    messages = [
        {
            "role": "system",
            "content": "Given a SQL insert query, you should predict a different name for a database table that is suitable to the information of the query. "
            "Answer only with the predicted name of the table. Don't give any explanation for your result.",
        },
        {
            "role": "user",
            "content": "Predict a different name of a database table for this insert query.\n"
            f"Query: {''.join(data_point['query'])}\n"
            "Table:",
        },
    ]

    for i in range(3):
        predicted_name = remove_quotes(model.run_prompt(messages))

        if (
            predicted_name != data_point["table_name"]
            and predicted_name
            not in synonyms[data_point["database_name"]][data_point["table_name"]]
        ):
            synonyms[data_point["database_name"]][data_point["table_name"]].append(
                predicted_name
            )


def generate_column_synonyms(
    data_point: dict[str, str], synonyms: dict[str, dict[str, dict[str, list[str]]]]
) -> None:
    if len(data_point["query"]) == 0:
        return

    for column in data_point["column_names"]:
        messages = [
            {
                "role": "system",
                "content": "Given three SQL insert query, you should predict a different name for the specified database column. "
                "Answer only with the different name of the column. Don't give any explanation for your result.",
            },
            {
                "role": "user",
                "content": f"Predict a different name for the database column '{column}' for these insert queries.\n"
                f"Query: {''.join(data_point['query'])}\n"
                "Table:",
            },
        ]

        for i in range(3):
            predicted_name = remove_quotes(model.run_prompt(messages))

            if (
                predicted_name != column
                and predicted_name
                not in synonyms[data_point["database_name"]][data_point["table_name"]][
                    column
                ]
            ):
                synonyms[data_point["database_name"]][data_point["table_name"]][
                    column
                ].append(predicted_name)


if __name__ == "__main__":
    with open(
        synonym_generation_data_path,
        encoding="utf-8",
    ) as synonym_generation_data_file:
        synonym_generation_data = json.load(synonym_generation_data_file)

    for data_point in tqdm(synonym_generation_data):
        if "column" in synonym_generation_data_path:
            synonyms = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
            generate_column_synonyms(data_point, synonyms)
        else:
            synonyms = defaultdict(lambda: defaultdict(lambda: []))
            generate_table_synonyms(data_point, synonyms)

    with open(
        synonyms_path,
        mode="w",
        encoding="utf-8",
    ) as synonyms_file:
        json.dump(synonyms, synonyms_file)
