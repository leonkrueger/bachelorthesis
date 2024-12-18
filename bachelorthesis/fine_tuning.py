# https://medium.com/@amodwrites/a-definitive-guide-to-qlora-fine-tuning-falcon-7b-with-peft-78f500a1f337
# https://medium.com/@ogbanugot/notes-on-fine-tuning-llama-2-using-qlora-a-detailed-breakdown-370be42ccca1

import os
from pathlib import Path

from system.utils.utils import get_finetuned_model_dir, load_env_variables

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
from trl import DataCollatorForCompletionOnlyLM

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
train_input_file = "missing_tables_12000_csv_columns_deleted"
validation_input_file = "missing_tables_csv_columns_deleted"
output_dir = "missing_tables_12000_1_csv_columns_deleted"
wandb_run_name = "12000_queries_1_epochs_csv_columns_deleted"

os.environ["WANDB_PROJECT"] = "bachelorthesis_missing_tables"


def generate_prompt(data_point):
    # Table prediction prompt:
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

    # Column prediction prompt (single column)
    # return [
    #     {
    #         "role": "system",
    #         "content": "You are an intelligent database that predicts the columns of a SQL-insert. "
    #         "The inserts can contain abbreviated or synonymous column names. The column names can also be missing entirely. "
    #         "Base your guess on the available information. "
    #         "If there is a suitable column in the table answer its name. Else, predict a suitable name for a new column in this table. "
    #         "Avoid answering with already predicted columns. "
    #         "Answer only with the name of the column. Don't give any explanation for your result.",
    #     },
    #     {
    #         "role": "user",
    #         "content": f"{data_point['Instruction']}\n" "Column:",
    #     },
    #     {"role": "assistant", "content": f"{data_point['Response'][8:]}"},
    # ]

    # Column prediction prompt (multiple columns)
    # return [
    #     {
    #         "role": "system",
    #         "content": "You are an intelligent database that predicts the columns of a SQL-insert. "
    #         "Predict the column name for each value in the insert. "
    #         "The inserts can contain abbreviated or synonymous column names. The column names can also be missing entirely. "
    #         "Base your guess on the available information. "
    #         "If there is a suitable column in the table use its name. Else, predict a suitable name for a new column in this table. "
    #         "Don't give any explanation for your result.",
    #     },
    #     {
    #         "role": "user",
    #         "content": f"{data_point['Instruction']}\n" "Columns:",
    #     },
    #     {"role": "assistant", "content": f"{data_point['Response'][9:]}"},
    # ]


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    return tokenizer.apply_chat_template(full_prompt, truncation=True, return_dict=True)


if __name__ == "__main__":
    wandb.login()
    wandb.init(name=wandb_run_name)

    # Load model and prepare for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=os.environ["HF_API_TOKEN"]
    )
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
    train_dataset_name = (
        Path(__file__)
        .resolve()
        .parent.joinpath(*os.environ["EVALUATION_BASE_DIR"].split("/"))
        / "fine_tuning"
        / "datasets"
        / f"{train_input_file}.json"
    )
    train_dataset = load_dataset("json", data_files=train_dataset_name, split="train")
    train_dataset = train_dataset.shuffle().map(generate_and_tokenize_prompt)

    # Create validation dataset
    validation_dataset_name = (
        Path(__file__)
        .resolve()
        .parent.joinpath(*os.environ["EVALUATION_BASE_DIR"].split("/"))
        / "fine_tuning"
        / "validation_datasets"
        / f"{validation_input_file}.json"
    )
    validation_dataset = load_dataset(
        "json", data_files=validation_dataset_name, split="train"
    )
    validation_dataset = validation_dataset.shuffle().map(generate_and_tokenize_prompt)

    output_dir = get_finetuned_model_dir(output_dir)
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
        output_dir=get_finetuned_model_dir("steps"),
        report_to="wandb",
        run_name=wandb_run_name,
    )

    # Train model
    response_template = "assistant"
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        args=training_args,
        data_collator=DataCollatorForCompletionOnlyLM(
            response_template, tokenizer=tokenizer
        ),
    )
    model.config.use_cache = False

    trainer.train()

    # Save model
    trainer.model.save_pretrained(output_dir)
