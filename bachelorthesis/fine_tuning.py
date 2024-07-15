# https://medium.com/@amodwrites/a-definitive-guide-to-qlora-fine-tuning-falcon-7b-with-peft-78f500a1f337
# https://medium.com/@ogbanugot/notes-on-fine-tuning-llama-2-using-qlora-a-detailed-breakdown-370be42ccca1

import os

from system.utils.utils import load_env_variables

load_env_variables()

import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
import wandb
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
train_input_file = "missing_columns_12000_combined_columns_2"
validation_input_file = "missing_columns_combined_columns_2"
output_dir = "missing_columns_12000_1_combined_columns_2"
wandb_run_name = "12000_queries_1_epochs_combined_columns_2"

os.environ["WANDB_PROJECT"] = "bachelorthesis_missing_columns"
wandb.login()
wandb.init(name=wandb_run_name)


def generate_prompt(data_point):
    # Table prediction prompt:
    # return [
    #     {
    #         "role": "system",
    #         "content": "You are an intelligent database that predicts on which table a SQL-insert should be executed. "
    #         "The inserts can contain abbreviated or synonymous names. The table and column names can be missing entirely. "
    #         "Base your guess on the available information. "
    #         "If there is a suitable table in the database answer its name. Else, predict a suitable name for a new database table. "
    #         "Answer only with the name of the table. Don't give any explanation for your result.",
    #     },
    #     {
    #         "role": "user",
    #         "content": f"{data_point['Instruction']}\n" "Table:",
    #     },
    #     {"role": "assistant", "content": f"{data_point['Response'][7:]}"},
    # ]

    # Column prediction prompt (single column)
    # return [
    #     {
    #         "role": "system",
    #         "content": "You are an intelligent database that predicts the columns of a SQL-insert. "
    #         "The inserts can contain abbreviated or synonymous column names. The column names can also be missing entirely. "
    #         "Base your guess on the available information. "
    #         "If there is a suitable column in the table answer its name. Else, predict a suitable name for a new column in this table. "
    #         "Answer only with the name of the column. Don't give any explanation for your result.",
    #     },
    #     {
    #         "role": "user",
    #         "content": f"{data_point['Instruction']}\n" "Column:",
    #     },
    #     {"role": "assistant", "content": f"{data_point['Response'][8:]}"},
    # ]

    # Column prediction prompt (multiple columns)
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
            "content": f"{data_point['Instruction']}\n" "Columns:",
        },
        {"role": "assistant", "content": f"{data_point['Response'][8:]}"},
    ]


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    return tokenizer.apply_chat_template(full_prompt, truncation=True, return_dict=True)


# Load model and prepare for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # load_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    token=os.environ["HF_API_TOKEN"],
)
model = prepare_model_for_kbit_training(model)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_API_TOKEN"])
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

# Create train dataset
train_dataset_name = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "..",
    "evaluation",
    "bachelorthesis",
    "fine_tuning",
    "datasets",
    f"{train_input_file}.json",
)
train_dataset = load_dataset("json", data_files=train_dataset_name, split="train")
train_dataset = train_dataset.shuffle().map(generate_and_tokenize_prompt)

# Create validation dataset
validation_dataset_name = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "..",
    "evaluation",
    "bachelorthesis",
    "fine_tuning",
    "validation_datasets",
    f"{validation_input_file}.json",
)
validation_dataset = load_dataset(
    "json", data_files=validation_dataset_name, split="train"
)
validation_dataset = validation_dataset.shuffle().map(generate_and_tokenize_prompt)

output_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "fine_tuning", "output", output_dir
)
os.makedirs(output_dir, exist_ok=True)

# Configure training arguments
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=64,
    eval_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",
    num_train_epochs=1,
    learning_rate=4e-4,
    fp16=True,
    logging_steps=1,
    evaluation_strategy="steps",
    eval_steps=10,
    save_strategy="no",
    output_dir=os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "fine_tuning", "output", "steps"
    ),
    report_to="wandb",
    run_name=wandb_run_name,
)

# Train model
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

trainer.train()

# Save model
trainer.model.save_pretrained(output_dir)
