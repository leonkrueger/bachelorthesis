import json
import os
import re
from typing import Any, Dict, List

from system.utils.utils import load_env_variables

load_env_variables()

from system.strategies.openai.openai import openai_execute
from tqdm import tqdm

# Switch if necessary
model = "gpt-3.5-turbo-0125"
evaluation_input_files = [
    "table_not_in_database",
    "table_in_database",
    "different_name_in_database",
]
evaluation_folder = os.path.join("further_evaluation", "performance_gpt_tables_deleted")

evaluation_base_folder = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    *os.environ["EVALUATION_BASE_DIR"].split("/"),
)
evaluation_folder = os.path.join(evaluation_base_folder, evaluation_folder)

max_tokens = 30


def generate_request(data_point) -> List[Dict[str, Any]]:
    return {
        "model": model,
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
                f"Query: {data_point['query']}\n"
                f"Database State:\n{data_point['database_state']}\n"
                "Table:",
            },
        ],
        "max_tokens": max_tokens,
    }


def extract_answer(response) -> str:
    return re.search(
        r"(?P<table>\S+)",
        (response["choices"][0]["message"]["content"]),
    ).group("table")


def run_experiments_for_strategy(
    evaluation_input: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    result_points = []

    requests = [generate_request(data_point) for data_point in evaluation_input]
    # print(
    #     "Estimated costs:",
    #     sum([_openai_compute_approximate_max_cost(request) for request in requests]),
    # )

    responses = openai_execute(requests)
    for data_point, response in zip(evaluation_input, responses):
        data_point["predicted_table_name"] = extract_answer(response)
        result_points.append(data_point)

    return result_points


for evaluation_input_file in tqdm(evaluation_input_files):
    with open(
        os.path.join(evaluation_folder, evaluation_input_file + ".json"),
        encoding="utf-8",
    ) as evaluation_file:
        evaluation_input = json.load(evaluation_file)

    output_folder = os.path.join(evaluation_folder, model)
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(
        output_folder, "results_" + evaluation_input_file + ".json"
    )

    # Continue if experiment was already run
    if os.path.exists(output_file_path):
        with open(output_file_path, encoding="utf-8") as results_file:
            if results_file.read().strip() != "":
                continue

    results = run_experiments_for_strategy(evaluation_input)

    with open(output_file_path, mode="w", encoding="utf-8") as output_file:
        json.dump(results, output_file)
