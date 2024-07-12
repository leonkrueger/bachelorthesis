import json
import os
from collections import defaultdict

from system.strategies.llama3.llama3_model import Llama3Model
from system.utils.utils import load_env_variables, remove_quotes
from tqdm import tqdm

load_env_variables()

model = Llama3Model()


def generate_synonyms(data_point: dict[str, str], synonyms) -> None:
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


with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "evaluation",
        "bachelorthesis",
        "fine_tuning",
        "synonym_generation_data.json",
    ),
    encoding="utf-8",
) as synonym_generation_data_file:
    synonym_generation_data = json.load(synonym_generation_data_file)

synonyms = defaultdict(lambda: defaultdict(lambda: []))

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
        "synonyms.json",
    ),
    mode="w",
    encoding="utf-8",
) as synonyms_file:
    json.dump(synonyms, synonyms_file)
