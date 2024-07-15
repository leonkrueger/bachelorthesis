import json
import os
from collections import defaultdict

from system.utils.utils import load_env_variables, remove_quotes

load_env_variables()

from system.strategies.llama3.llama3_model import Llama3Model
from tqdm import tqdm

model = Llama3Model()


def generate_synonyms(data_point: dict[str, str], synonyms) -> None:
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


with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "evaluation",
        "bachelorthesis",
        "fine_tuning",
        "column_synonym_generation_data.json",
    ),
    encoding="utf-8",
) as synonym_generation_data_file:
    synonym_generation_data = json.load(synonym_generation_data_file)

synonyms = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

for data_point in tqdm(synonym_generation_data):
    generate_synonyms(data_point, synonyms)

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "evaluation",
        "bachelorthesis",
        "fine_tuning",
        "column_synonyms.json",
    ),
    mode="w",
    encoding="utf-8",
) as synonyms_file:
    json.dump(synonyms, synonyms_file)
