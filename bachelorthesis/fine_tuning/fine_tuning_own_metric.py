# https://medium.com/@amodwrites/a-definitive-guide-to-qlora-fine-tuning-falcon-7b-with-peft-78f500a1f337
# https://medium.com/@ogbanugot/notes-on-fine-tuning-llama-2-using-qlora-a-detailed-breakdown-370be42ccca1

import os

import bitsandbytes as bnb
import numpy as np
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

np.set_printoptions(threshold=np.inf)
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


def generate_prompt(data_point, validation_set):
    messages = [
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
            "content": f"{data_point['Instruction']}\nTable:",
        },
        # {"role": "assistant", "content": f"{data_point['Response'][7:]}"},
    ]
    if not validation_set:
        messages.append(
            {"role": "assistant", "content": f"{data_point['Response'][7:]}"}
        )
    return messages


def generate_and_tokenize_prompt(data_point, validation_set):
    full_prompt = generate_prompt(data_point, validation_set)
    inputs = tokenizer.apply_chat_template(
        full_prompt, truncation=True, return_dict=True
    )
    output = tokenizer(data_point["Response"])
    inputs["labels"] = output["input_ids"]
    del inputs["attention_mask"]
    if validation_set:
        print(inputs)
    return inputs


def compute_metrics(predictions) -> dict[str, float]:
    labels = predictions.label_ids
    preds = predictions.predictions  # .argmax(-1)
    # if isinstance(preds, tuple):
    #     preds = preds[0]
    print("Preds[0]:", preds[0])
    print("Preds[1]:", preds[1])
    print("Labels:", labels)
    test = np.where(preds[0] != -100, preds[0], tokenizer.pad_token_id)
    decoded_test = tokenizer.batch_decode([test], skip_special_tokens=True)
    print("Decoded Preds[0]:", decoded_test)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds[1] != -100, preds[1], tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print("Decoded Labels:", decoded_labels)
    print("Decoded Preds:", decoded_preds)

    accuracy = len(
        [pred for pred, label in zip(decoded_preds, decoded_labels) if pred == label]
    ) / len(preds)
    wandb.log({"eval/accuracy": accuracy})
    return {"accuracy": accuracy}


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


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
train_dataset = train_dataset.shuffle().map(
    lambda data_point: generate_and_tokenize_prompt(data_point, False)
)

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
validation_dataset = (
    validation_dataset.shuffle()
    .take(1)
    .map(lambda data_point: generate_and_tokenize_prompt(data_point, True))
)

output_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "output", output_dir
)
os.makedirs(output_dir, exist_ok=True)

# Configure training arguments
training_args = transformers.Seq2SeqTrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=64,
    eval_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=4e-4,
    fp16=True,
    logging_steps=1,
    evaluation_strategy="steps",
    eval_steps=1,
    predict_with_generate=True,
    # generation_max_length=30,
    save_strategy="no",
    output_dir=os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "output", "steps"
    ),
    report_to="wandb",
    run_name=wandb_run_name,
)

# Train model
trainer = transformers.Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

trainer.train()

# Save model
trainer.model.save_pretrained(output_dir)
