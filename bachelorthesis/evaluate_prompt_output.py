import copy
import json
import os
import re
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

# database = PythonDatabase()
# table_manager = TableManager(database)

HF_API_TOKEN = "YOUR_HF_API_TOKEN"

# Switch if necessary
strategy_name = "missing_tables_1500"
evaluation_input_file = "table_not_in_database.json"
evaluation_folder = os.path.join("further_evaluation", "error_cases_missing_tables")

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
max_new_tokens = 10

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_API_TOKEN)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=HF_API_TOKEN,
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    base_model,
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "fine_tuning",
        "output",
        strategy_name,
    ),
)
model = model.merge_and_unload()

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


def generate_prompt(data_point):
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
            f"Database State:\n{data_point['database_state']}\n"
            "Table:",
        },
    ]


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    return tokenizer.apply_chat_template(
        full_prompt, tokenize=False, add_generation_prompt=True
    )


def run_prompt(prompt) -> str:
    return re.search(
        r"(?P<table>\S+)",
        pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )[0]["generated_text"][len(prompt) :].strip(),
    ).group("table")


def run_experiments_for_strategy(
    evaluation_input: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    result_points = []
    for data_point in evaluation_input:
        # Run prompt directly
        # prompt = generate_and_tokenize_prompt(data_point)
        # data_point["predicted_table_name"] = run_prompt(prompt)
        # result_points.append(data_point)

        # Get different table name first
        result_point = copy.deepcopy(data_point)
        data_point["query"] = (
            data_point["query"][:12]
            + data_point["query"][data_point["query"].find("(") :]
        )
        table_name_generation_prompt = generate_and_tokenize_prompt(data_point)
        table_name = run_prompt(table_name_generation_prompt)

        if table_name == data_point["expected_table_name"]:
            continue

        result_point["database_state"][table_name] = result_point["database_state"].pop(
            result_point["expected_table_name"]
        )
        result_point["expected_table_name"] == table_name

        # Run evaluation prompt
        prompt = generate_and_tokenize_prompt(result_point)
        result_point["predicted_table_name"] = run_prompt(prompt)
        result_points.append(result_point)

    return result_points


with open(
    os.path.join(evaluation_folder, evaluation_input_file), encoding="utf-8"
) as evaluation_file:
    evaluation_input = json.load(evaluation_file)

output_folder = os.path.join(evaluation_folder, strategy_name)
output_file_path = os.path.join(output_folder, "results_" + evaluation_input_file)

# Continue if experiment was already run
with open(output_file_path, encoding="utf-8") as results_file:
    if results_file.read().strip() != "":
        exit()

results = run_experiments_for_strategy(evaluation_input)

with open(output_file_path, mode="w", encoding="utf-8") as output_file:
    json.dump(results, output_file)

errors_file.close()
os.chmod(errors_file_path, 0o777)
