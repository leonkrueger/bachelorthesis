import copy
import json
import os
import re
from typing import Any, Dict, List

import torch
from peft import PeftModel
from system.data.query_data import QueryData
from system.insert_query_parser import parse_insert_query
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from utils import load_env_variables

load_env_variables()

# Switch if necessary
strategy_name = "missing_columns_12000_combined_columns_2"
fine_tuned_model_folder = "missing_columns_12000_1_combined_columns_2"
evaluation_input_files = [
    "evaluation_data",
    "evaluation_data_columns_deleted",
]
evaluation_folder = os.path.join(
    "further_evaluation", "error_cases_missing_columns_combined_columns"
)
different_name_already_generated = True

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
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
max_new_tokens = 300

tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_API_TOKEN"])
if fine_tuned_model_folder:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.environ["HF_API_TOKEN"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        base_model,
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "fine_tuning",
            "output",
            fine_tuned_model_folder,
        ),
    )
    model = model.merge_and_unload()
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.environ["HF_API_TOKEN"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

tokenizer.pad_token = tokenizer.eos_token
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)
terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]


def generate_prompt_for_single_column(data_point, value, column="No column specified"):
    return [
        {
            "role": "system",
            "content": "You are an intelligent database that predicts the columns of a SQL-insert. "
            "The inserts can contain abbreviated or synonymous column names. The column names can also be missing entirely. "
            "Base your guess on the available information. "
            "If there is a suitable column in the table answer its name. Else, predict a suitable name for a new column in this table. "
            "Answer only with the name of the column. Don't give any explanation for your result.",
        },
        {
            "role": "user",
            "content": (
                "Predict the column for this value:\n"
                f"Specified column: {column}\n"
                f"Value: {value}\n"
                f"{data_point['table_state']}\n"
                "Column:",
            ),
        },
    ]


def generate_prompt(data_point):
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
            "content": "Predict the columns for this query:\n"
            f"Query: {data_point['query']}\n"
            f"{data_point['table_state']}\n"
            "Columns:",
        },
    ]


def generate_and_tokenize_prompt_for_single_column(data_point, value, column):
    full_prompt = generate_prompt_for_single_column(data_point, value, column)
    return tokenizer.apply_chat_template(
        full_prompt, tokenize=False, add_generation_prompt=True
    )


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    return tokenizer.apply_chat_template(
        full_prompt, tokenize=False, add_generation_prompt=True
    )


def run_prompt(prompt) -> str:
    # return re.search(
    #     r"(?P<column>\S+)",
    #     pipe(
    #         prompt,
    #         max_new_tokens=max_new_tokens,
    #         eos_token_id=terminators,
    #         do_sample=True,
    #         temperature=0.6,
    #         top_p=0.9,
    #     )[0]["generated_text"][len(prompt) :].strip(),
    # ).group("column")
    return pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )[0]["generated_text"][len(prompt) :].strip()


def run_experiments_for_strategy(
    evaluation_input: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    result_points = []
    for data_point in tqdm(evaluation_input):
        # Run prompt directly
        if "combined_columns" in fine_tuned_model_folder:
            prompt = generate_and_tokenize_prompt(data_point)
            data_point["predicted_column_names"] = run_prompt(prompt)
        else:
            query_data = parse_insert_query(QueryData(data_point["query"], None))
            if not query_data.columns:
                query_data.columns = [None for i in range(len(query_data.values[0]))]
            data_point["predicted_column_names"] = ";".join(
                [
                    run_prompt(
                        generate_and_tokenize_prompt_for_single_column(
                            data_point, value, column
                        )
                    )
                    for value, column in zip(query_data.values[0], query_data.columns)
                ]
            )
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
