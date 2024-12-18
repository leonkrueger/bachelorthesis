import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from system.utils.utils import get_finetuned_model_dir, load_env_variables

load_env_variables()

from system.strategies.llama3.llama3_model import Llama3Model
from tqdm import tqdm

# Switch if necessary
strategy_name = "missing_tables_1500"
fine_tuned_model_folder = "missing_tables_1500_1"
evaluation_input_files = [
    "table_not_in_database",
    "table_in_database",
    "different_name_in_database",
]
evaluation_folder = Path("further_evaluation", "error_cases_missing_tables")
different_name_already_generated = True

evaluation_base_folder = (
    Path(__file__)
    .resolve()
    .parent.joinpath(*os.environ["EVALUATION_BASE_DIR"].split("/"))
)
evaluation_folder = evaluation_base_folder / evaluation_folder

# Create model
model = Llama3Model(get_finetuned_model_dir(fine_tuned_model_folder))


def generate_prompt(data_point):
    database_string = (
        data_point["database_state"]
        if isinstance(data_point["database_state"], str)
        else (
            "\n".join(
                [
                    f"- Table: {table}, Columns: [{', '.join([column[0] for column in columns])}]"
                    for table, columns in data_point["database_state"].items()
                ]
            )
            if len(data_point["database_state"]) > 0
            else "No table exists yet."
        )
    )

    return [
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
            f"Database State:\n{database_string}\n"
            "Table:",
        },
    ]


def run_experiments_for_strategy(
    evaluation_input: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    result_points = []
    for data_point in tqdm(evaluation_input):
        if (
            not different_name_already_generated
            and evaluation_input_file == "different_name_in_database"
        ):
            # Get different table name first
            result_point = copy.deepcopy(data_point)
            data_point["query"] = (
                data_point["query"][:12]
                + data_point["query"][data_point["query"].find("(") :]
            )
            table_name_generation_prompt = generate_prompt(data_point)

            database_tables = (
                [
                    match[6:-1]
                    for match in re.findall(r"Table \S+", data_point["database_state"])
                ]
                if isinstance(data_point["database_state"], str)
                else data_point["database_state"].keys()
            )

            # Try 3 times. When no alternative name is found, skip this insert.
            for i in range(3):
                table_name = model.run_prompt(table_name_generation_prompt)
                if table_name not in database_tables:
                    break
            if table_name in database_tables:
                continue

            if isinstance(result_point["database_state"], str):
                result_point["database_state"] = result_point["database_state"].replace(
                    f"Table {result_point['expected_table_name']}:",
                    f"Table {table_name}:",
                )
            else:
                result_point["database_state"][table_name] = result_point[
                    "database_state"
                ].pop(result_point["expected_table_name"])
            result_point["expected_table_name"] = table_name

            # Run evaluation prompt
            prompt = generate_prompt(result_point)
            result_point["predicted_table_name"] = model.run_prompt(prompt)
            result_points.append(result_point)
        else:
            # Run prompt directly
            prompt = generate_prompt(data_point)
            data_point["predicted_table_name"] = model.run_prompt(prompt)
            result_points.append(data_point)

    return result_points


if __name__ == "__main__":
    for evaluation_input_file in tqdm(evaluation_input_files):
        with open(
            evaluation_folder / (evaluation_input_file + ".json"),
            encoding="utf-8",
        ) as evaluation_file:
            evaluation_input = json.load(evaluation_file)

        output_folder = evaluation_folder / strategy_name
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = output_folder / (
            "results_" + evaluation_input_file + ".json"
        )

        # Continue if experiment was already run
        if os.path.exists(output_file_path):
            with open(output_file_path, encoding="utf-8") as results_file:
                if results_file.read().strip() != "":
                    continue

        results = run_experiments_for_strategy(evaluation_input)

        with open(output_file_path, mode="w", encoding="utf-8") as output_file:
            json.dump(results, output_file)
