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
        predicted_name = remove_quotes(run_prompt(messages))

        if (
            predicted_name != data_point["table_name"]
            and predicted_name
            not in synonyms[data_point["database_name"]]["table_name"]
        ):
            synonyms[data_point["database_name"]]["table_name"].append(predicted_name)


with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "evaluation",
        "bachelorthesis",
        "fine_tuning",
        "synonym_generation_data.json",
    )
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
        "..",
        "evaluation",
        "bachelorthesis",
        "fine_tuning",
        "synonyms.json",
    )
) as synonyms_file:
    json.dump(synonyms, synonyms_file)
