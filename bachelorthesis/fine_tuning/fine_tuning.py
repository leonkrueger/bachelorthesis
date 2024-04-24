# https://medium.com/@amodwrites/a-definitive-guide-to-qlora-fine-tuning-falcon-7b-with-peft-78f500a1f337
# https://medium.com/@ogbanugot/notes-on-fine-tuning-llama-2-using-qlora-a-detailed-breakdown-370be42ccca1

import os

import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


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


HF_API_TOKEN = "YOUR_HF_API_TOKEN"

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
input_file = "missing_tables_300"

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
dataset_name = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "input", f"{input_file}.json"
)
dataset = load_dataset("json", data_files=dataset_name, split="train")
dataset = dataset.shuffle().map(generate_and_tokenize_prompt)

output_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "output", input_file
)
os.makedirs(output_dir, exist_ok=True)

# Configure training arguments
training_args = transformers.TrainingArguments(
    auto_find_batch_size=True,
    num_train_epochs=1,  # TODO
    # learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_total_limit=1,  # TODO
    save_strategy="epoch",
    output_dir=os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "output", "steps"
    ),
)

# Train model
trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()

# Save model
trainer.model.save_pretrained(output_dir)
