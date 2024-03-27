import os
import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

model_id = "tiiuae/falcon-7b"  # "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)


def generate_prompt(data_point):
    return f"""
<Human>: {data_point["Context"]}
<AI>: {data_point["Response"]}
  """.strip()


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
    return tokenized_full_prompt


from datasets import load_dataset

dataset_name = "Amod/mental_health_counseling_conversations"
dataset = load_dataset(dataset_name, split="train")

dataset = dataset.shuffle().map(generate_and_tokenize_prompt)

training_args = transformers.TrainingArguments(
    auto_find_batch_size=True,
    num_train_epochs=4,
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=4,
    logging_steps=10,
    output_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), "output"),
    save_strategy="epoch",
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()
