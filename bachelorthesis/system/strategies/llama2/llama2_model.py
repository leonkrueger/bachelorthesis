import re
import time
from enum import Enum

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


class LLama2ModelType(Enum):
    FINE_TUNED = "fine_tuned"
    NON_FINE_TUNED_LOCAL = "non_fine_tuned_local"
    NON_FINE_TUNED_API = "non_fine_tuned_api"


class LLama2Model(Strategy):
    def __init__(
        self,
        model_type: LLama2ModelType = LLama2ModelType.NON_FINE_TUNED_LOCAL,
        fine_tuned_model_dir: str = None,
        huggingface_api_token: str = None,
    ) -> None:
        self.model_name = "meta-llama/Llama-2-7b-hf"
        self.model_type = model_type
        self.max_new_tokens = 10

        if model_type == LLama2ModelType.NON_FINE_TUNED_API:
            self.client = InferenceClient(token=huggingface_api_token, timeout=300)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=huggingface_api_token
            )
            if model_type == LLama2ModelType.NON_FINE_TUNED_LOCAL:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, token=huggingface_api_token, device_map="auto"
                )
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    load_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    token=huggingface_api_token,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
                model = PeftModel.from_pretrained(base_model, fine_tuned_model_dir)
                model = model.merge_and_unload()

            # tokenizer.pad_token = tokenizer.unk_token
            # model.config.pad_token_id = tokenizer.pad_token_id
            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                do_sample=True,
                max_new_tokens=self.max_new_tokens,
                top_k=5,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

    def run_prompt(self, prompt_text: str) -> str:
        """Runs a prompt on a Llama2 model and returns its answer"""
        if self.model_type == LLama2ModelType.NON_FINE_TUNED_API:
            time.sleep(5)
            return self.client.text_generation(
                prompt_text,
                max_new_tokens=self.max_new_tokens,
                model=self.model_name,
            ).strip()
        else:
            return self.pipe(prompt_text)[0]["generated_text"]

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
            r"Table: (?P<table>\S+)",
            self.run_prompt(
                "<s>[INST] <<SYS>>\n"
                "You are an intelligent database that predicts on which table a SQL-insert should be executed. "
                "The inserts can contain abbreviated or synonymous names. The table and column names can be missing entirely. "
                "Base your guess on the available information. "
                # "You give the output in the form 'Table: {table_name}'. If there is a suitable table in the database, "
                # "you replace '{table_name}' with its name. Else, you replace '{table_name}' with a suitable name for a database table. "
                "If there is a suitable table in the database answer its name. Else, predict a suitable name for a new database table. "
                "Answer only with the name of the table. Don't give any explanation for your result.\n"
                "<</SYS>>\n"
                # "Predict the table for this example:\n"
                f"Query: {query_data.query}\n"
                f"Database State:\n{database_string}[/INST]\n"
                "Table:",
            ),
        ).group("table")
