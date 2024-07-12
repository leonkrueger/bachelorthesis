import os
from typing import List

import torch
from huggingface_hub import InferenceClient
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ...data.query_data import QueryData


class NamePredictor:
    def __init__(self) -> None:
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.max_new_tokens = 10

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=os.environ["HF_API_TOKEN"]
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=os.environ["HF_API_TOKEN"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer.pad_token = tokenizer.eos_token
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )

    def run_prompt(self, messages: list[dict[str, str]]) -> str:
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
            max_new_tokens=10,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )[0]["generated_text"][len(prompt) :].strip()

    def remove_quotes(name: str) -> str:
        """Removes the quotes from a name, if it has any"""
        if name.startswith("'") or name.startswith('"'):
            return name[1:-1]
        return name

    def predict_table_name(self, query_data: QueryData) -> str:
        """Predicts a suitable table name for an insertion query"""
        messages = [
            {
                "role": "system",
                "content": "Given a SQL insert query, you should predict a name for a database table that is suitable to the information of the query. "
                "Answer only with the predicted name of the table. Don't give any explanation for your result.",
            },
            {
                "role": "user",
                "content": "Predict a name for a database table for this insert query.\n"
                f"Query: {''.join(query_data.query)}\n"
                "Table:",
            },
        ]

        return self.remove_quotes(self.run_prompt(messages))

    def predict_column_name(self, query_data: QueryData, value: str) -> str:
        """Predicts a suitable column name for a specific value of a query"""
        messages = [
            {
                "role": "system",
                "content": "Given a SQL insert query and a specific value, you should predict a name for a database column "
                "that is suitable to store the specified value of the query. "
                "Answer only with the predicted name of the column. Don't give any explanation for your result.",
            },
            {
                "role": "user",
                "content": "Predict a name for a database column for this value.\n"
                f"Query: {''.join(query_data.query)}\n"
                f"Value: {value}"
                "Column:",
            },
        ]

        return self.remove_quotes(self.run_prompt(messages))
