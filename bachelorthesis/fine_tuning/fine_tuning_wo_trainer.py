import os

import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data.dataloader import DataLoader
from transformers import (
    Adam,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

HF_API_TOKEN = "YOUR_HF_API_TOKEN"

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
train_input_file = "missing_tables_12000"
validation_input_file = "missing_tables"
output_dir = "missing_tables_12000_1"
wandb_run_name = "12000_queries_1_epochs"

os.environ["WANDB_PROJECT"] = "bachelorthesis_missing_tables"
wandb.login()
wandb.init(name=wandb_run_name)
# wandb.define_metric("eval/accuracy", summary="min")


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
            "content": f"{data_point['Instruction']}\n" "Table:",
        },
        {"role": "assistant", "content": f"{data_point['Response'][7:]}"},
    ]


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    return tokenizer.apply_chat_template(full_prompt, return_dict=True)


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
    token=HF_API_TOKEN,
)
model = prepare_model_for_kbit_training(model)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_API_TOKEN)
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
    os.path.dirname(os.path.realpath(__file__)), "output", output_dir
)
os.makedirs(output_dir, exist_ok=True)

training_args = {
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,  #
    "gradient_accumulation_steps": 64,
    "eval_accumulation_steps": 8,  #
    "optim": "adamw_bnb_8bit",
    "num_train_epochs": 1,  #
    "learning_rate": 4e-4,
    "fp16": True,
    "logging_steps": 1,  #
    "evaluation_strategy": "steps",  #
    "eval_steps": 10,  #
    "save_strategy": "no",  #
    "output_dir": os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "output", "steps"
    ),  #
    "report_to": "wandb",  #
    "run_name": wandb_run_name,  #
}

dataloader = DataLoader(
    train_dataset, batch_size=training_args.per_device_train_batch_size
)
optim = bnb.optim.Adam8bit(model.parameters(), training_args.learning_rate)

if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

accelerator = Accelerator(fp16=training_args.fp16)
model, optimizer, dataloader = accelerator.prepare(model, optim, dataloader)

model.train()
accumulated_loss = 0
for step, batch in enumerate(dataloader, start=1):
    loss = model(**batch).loss
    loss = loss / training_args.gradient_accumulation_steps
    accumulated_loss += loss
    accelerator.backward(loss)
    if step % training_args.gradient_accumulation_steps == 0:
        wandb.log(
            {"train/loss": accumulated_loss},
            step=int(step / training_args.gradient_accumulation_steps),
        )
        optimizer.step()
        optimizer.zero_grad()
