import json
import os
from typing import Any, Dict, List, Tuple

from system.utils.utils import load_env_variables

load_env_variables()

from system.data.query_data import QueryData
from system.insert_query_parser import parse_insert_query
from system.strategies.llama3.llama3_model import Llama3Model
from tqdm import tqdm

# Switch if necessary
strategy_name = "missing_columns_12000_csv_already_predicted"
fine_tuned_model_folder = "missing_columns_12000_1_csv_already_predicted"
evaluation_input_files = [
    "evaluation_data",
    "evaluation_data_columns_deleted",
]
evaluation_folder = os.path.join(
    "further_evaluation", "error_cases_missing_columns_combined_columns"
)
different_name_already_generated = True
max_new_tokens = 30

# Depends on if the script is run in Docker or as plain python
# evaluation_base_folder = os.path.join("/app", "evaluation")
evaluation_base_folder = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "..",
    "evaluation",
    "bachelorthesis",
)
evaluation_folder = os.path.join(evaluation_base_folder, evaluation_folder)

errors_file_path = os.path.join(evaluation_base_folder, "errors.txt")
errors_file = open(errors_file_path, "w", encoding="utf-8")

# Create model
model = Llama3Model(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "fine_tuning",
        "output",
        fine_tuned_model_folder,
    )
)


def generate_prompt_for_single_column(
    data_point,
    value,
    column="No column specified",
    already_predicted_columns=None,
    custom_table_state=None,
):
    return [
        {
            "role": "system",
            "content": (
                "You are an intelligent database that predicts the columns of a SQL-insert. "
                "The inserts can contain abbreviated or synonymous column names. The column names can also be missing entirely. "
                "Base your guess on the available information. "
                "If there is a suitable column in the table answer its name. Else, predict a suitable name for a new column in this table. "
                + "Avoid answering with already predicted columns. "
                if already_predicted_columns is not None
                else ""
                + "Answer only with the name of the column. Don't give any explanation for your result."
            ),
        },
        {
            "role": "user",
            "content": (
                (
                    (
                        "Predict the column for this value:\n"
                        f"Query: {data_point['query']}\n"
                        f"{data_point['table_state']}\n"
                        f"Already predicted columns: {', '.join(already_predicted_columns)}\n"
                        f"Specified column: {column}\n"
                        f"Value: {value}\n"
                        "Column:"
                    ),
                )
                if already_predicted_columns is not None
                else (
                    "Predict the column for this value:\n"
                    f"Specified column: {column}\n"
                    f"Value: {value}\n"
                    f"{custom_table_state if custom_table_state is not None else data_point['table_state']}\n"
                    "Column:"
                )
            ),
        },
    ]


def generate_prompt(data_point, number_of_columns):
    return [
        {
            "role": "system",
            "content": "You are an intelligent database that predicts the columns of a SQL-insert. "
            "Predict the column name for each value in the insert. "
            "The inserts can contain abbreviated or synonymous column names. The column names can also be missing entirely. "
            "Base your guess on the available information. "
            "If there is a suitable column in the table use its name. Else, predict a suitable name for a new column in this table. "
            "Don't give any explanation for your result.",
        },
        {
            "role": "user",
            "content": f"Predict the {number_of_columns} columns for this query:\n"
            f"Query: {data_point['query']}\n"
            f"{data_point['table_state']}\n"
            "Columns:",
        },
    ]


def get_table_state_from_str(
    table_state: str,
) -> Tuple[str, List[str], List[List[str]]]:
    rows = table_state.split("\n")
    table = rows[0][6:-1]
    columns = rows[1].split(";")
    values = (
        []
        if len(rows) <= 2
        else [[value for value in row.split(";")] for row in rows[2:]]
    )
    return (table, columns, values)


def get_table_state_str(
    table_name: str, columns: List[str], values: List[List[str]]
) -> str:
    return f"Table {table_name}:\n{';'.join(columns)}\n" + "\n".join(
        [";".join(row) for row in values]
    )


def run_experiments_for_strategy(
    evaluation_input: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    result_points = []
    for data_point in tqdm(evaluation_input):
        query_data = parse_insert_query(QueryData(data_point["query"], None))
        # Run prompt directly
        if "combined_columns" in fine_tuned_model_folder:
            prompt = generate_prompt(data_point, len(query_data.values[0]))
            data_point["predicted_column_names"] = model.run_prompt(
                prompt, max_new_tokens
            )
        else:
            if not query_data.columns:
                query_data.columns = [None for i in range(len(query_data.values[0]))]

            data_point["predicted_column_names"] = []
            table, columns, values = get_table_state_from_str(data_point["table_state"])

            # Check that the data was parsed correctly
            # If not, ignore this data point
            if len(values) != 0 and (
                len(columns) != len(values[0])
                or any([len(row) != len(values[0]) for row in values[1:]])
            ):
                continue

            for value, column in zip(query_data.values[0], query_data.columns):
                predicted = model.run_prompt(
                    generate_prompt_for_single_column(
                        data_point,
                        value,
                        column,
                        custom_table_state=get_table_state_str(table, columns, values),
                    ),
                    max_new_tokens,
                )
                data_point["predicted_column_names"].append(predicted)

                # Changes table state, so that predicted column is no longer included
                if predicted in columns:
                    column_index = columns.index(predicted)
                    columns.remove(predicted)
                    values = [
                        [value for i, value in enumerate(row) if i != column_index]
                        for row in values
                    ]
        result_points.append(data_point)

    return result_points


for evaluation_input_file in tqdm(evaluation_input_files):
    with open(
        os.path.join(evaluation_folder, evaluation_input_file + ".json"),
        encoding="utf-8",
    ) as evaluation_file:
        evaluation_input = json.load(evaluation_file)

    output_folder = os.path.join(evaluation_folder, strategy_name)
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

errors_file.close()
os.chmod(errors_file_path, 0o777)
