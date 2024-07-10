import os
import re
import time
from enum import Enum
from typing import Dict, List

import torch
from huggingface_hub import InferenceClient
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from ...data.query_data import QueryData
from ..strategy import Strategy


class Llama3ModelType(Enum):
    FINE_TUNED = "fine_tuned"
    NON_FINE_TUNED = "non_fine_tuned"


class Llama3Model(Strategy):
    def __init__(
        self,
        model_type: Llama3ModelType = Llama3ModelType.NON_FINE_TUNED,
        fine_tuned_model_dir: str = None,
    ) -> None:
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.model_type = model_type
        self.max_new_tokens = 30

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=os.environ["OPENAI_API_KEY"]
        )
        if model_type == Llama3ModelType.NON_FINE_TUNED:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=os.environ["OPENAI_API_KEY"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=os.environ["OPENAI_API_KEY"],
                quantization_config=bnb_config,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(base_model, fine_tuned_model_dir)
            model = model.merge_and_unload()

        tokenizer.pad_token = tokenizer.eos_token
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )

    def run_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Runs a prompt on a Llama2 model and returns its answer"""
        prompt = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        terminators = [
            self.pipe.tokenizer.eos_token_id,
            self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        return self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )[0]["generated_text"][len(prompt) :].strip()

    def predict_table_name(self, query_data: QueryData) -> str:
        database_string = (
            "\n".join(
                [
                    f"- Table: {table}, Columns: [{', '.join([column[0] for column in columns])}]"
                    for table, columns in query_data.database_state.items()
                ]
            )
            if len(query_data.database_state) > 0
            else "No table exists yet."
        )

        return re.search(
            r"(?P<table>\S+)",
            self.run_prompt(
                [
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
                        f"Query: {query_data.get_query(use_quotes=False)}\n"
                        f"Database State:\n{database_string}\n"
                        "Table:",
                    },
                ]
            ),
        ).group("table")
