"""
Evaluate individual prompt outputs for the column mapping task by a (fine-tuned) LLM.

Stores the results in {evaluation_base_folder}/{evaluation_folder}/{strategy_name}

``fine_tuned_model_folder`` specifies the name of the fine-tuned model
``evaluation_input_files`` contain the input for this evaluation (need to be in {evaluation_base_folder}/{evaluation_folder})
``max_new_tokens`` specifies the number of tokens each output can contain
``model`` specifies the LLM that is used
"""

import json
import os
from pathlib import Path
from typing import Any

from system.utils.utils import get_finetuned_model_dir, load_env_variables

load_env_variables()

from system.data.insert_data import InsertData
from system.insert_parser import parse_insert
from system.strategies.llama3.llama3_model import Llama3Model
from tqdm import tqdm

# Switch if necessary
strategy_name = (
    "missing_columns_12000_own_data_collator_predicted_removed_from_table_state"
)
fine_tuned_model_folder = "missing_columns_12000_1_own_data_collator"
evaluation_input_files = [
    "evaluation_data",
    "evaluation_data_columns_deleted",
]
evaluation_folder = Path(
    "further_evaluation", "error_cases_missing_columns_combined_columns"
)
max_new_tokens = 30

evaluation_base_folder = (
    Path(__file__)
    .resolve()
    .parent.joinpath(*os.environ["EVALUATION_BASE_DIR"].split("/"))
)
evaluation_folder = evaluation_base_folder / evaluation_folder

# Create model
model = Llama3Model(get_finetuned_model_dir(fine_tuned_model_folder))


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
                    f"Query: {data_point['query']}\n"
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


def is_usable_value(value: str | Any) -> bool:
    return value is not None and value.lower() != "'nan'" and value.lower() != "null"


def get_table_state_from_str(
    table_state: str,
) -> tuple[str, list[str], list[list[str]]]:
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
    table_name: str, columns: list[str], values: list[list[str]]
) -> str:
    # return f"Table {table_name}:\n{';'.join(columns)}\n" + "\n".join(
    #     [";".join(row) for row in values]
    # )
    return f"Table {table_name}:\n" + "\n".join(
        [
            f"Column {column_name}, Example values: [{', '.join([row[column_index] for row in values if is_usable_value(row[column_index])])}]"
            for column_index, column_name in enumerate(columns)
        ]
    )


def run_experiments_for_strategy(
    evaluation_input: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result_points = []
    for data_point in tqdm(evaluation_input):
        insert_data = parse_insert(InsertData(data_point["query"], None))

        if not insert_data.columns:
            insert_data.columns = [None for i in range(len(insert_data.values[0]))]

        # Run prompt directly
        if "combined_columns" in strategy_name:
            prompt = generate_prompt(data_point, len(insert_data.values[0]))
            data_point["predicted_column_names"] = model.run_prompt(
                prompt, max_new_tokens
            )
        elif "predicted_removed_from_table_state" in strategy_name:
            data_point["predicted_column_names"] = []
            table, columns, values = get_table_state_from_str(data_point["table_state"])

            # Check that the data was parsed correctly
            # If not, ignore this data point
            if len(values) != 0 and (
                len(columns) != len(values[0])
                or any([len(row) != len(values[0]) for row in values[1:]])
            ):
                continue

            for value, column in zip(insert_data.values[0], insert_data.columns):
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
        elif "already_predicted" in strategy_name:
            predicted_columns = []
            for value, column in zip(insert_data.values[0], insert_data.columns):
                predicted_columns.append(
                    model.run_prompt(
                        generate_prompt_for_single_column(
                            data_point,
                            value,
                            column,
                            already_predicted_columns=predicted_columns,
                        ),
                        max_new_tokens,
                    )
                )
            data_point["predicted_column_names"] = predicted_columns
        else:
            data_point["predicted_column_names"] = [
                model.run_prompt(
                    generate_prompt_for_single_column(data_point, value, column),
                    max_new_tokens,
                )
                for value, column in zip(insert_data.values[0], insert_data.columns)
            ]
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
        if output_file_path.exists():
            with open(output_file_path, encoding="utf-8") as results_file:
                if results_file.read().strip() != "":
                    continue

        results = run_experiments_for_strategy(evaluation_input)

        with open(output_file_path, mode="w", encoding="utf-8") as output_file:
            json.dump(results, output_file)
