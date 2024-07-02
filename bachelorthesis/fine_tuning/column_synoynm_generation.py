import gc
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

HF_API_TOKEN = "YOUR_HF_API_TOKEN"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_API_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=HF_API_TOKEN,
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


def run_prompt(messages: list[dict[str, str]]) -> str:
    """Runs a prompt on a Llama2 model and returns its answer"""
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    return pipe(
        prompt,
        max_new_tokens=10,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )[0]["generated_text"][len(prompt) :].strip()


def remove_quotes(name: str) -> str:
    if name.startswith("'") or name.startswith('"'):
        return name[1:-1]
    return name


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
            predicted_name = remove_quotes(run_prompt(messages))

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
