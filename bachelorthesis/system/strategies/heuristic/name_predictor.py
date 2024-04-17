from typing import List

import torch
from huggingface_hub import InferenceClient
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

USE_LOCAL_MODEL = True


class NamePredictor:
    def __init__(self, HF_API_TOKEN: str = "") -> None:
        print(torch.cuda.is_available())
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        self.max_new_tokens = 1

        if USE_LOCAL_MODEL:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="auto"
            )

            tokenizer.pad_token = tokenizer.unk_token
            model.config.pad_token_id = tokenizer.pad_token_id
            pipe = pipeline(
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
            self.llm = HuggingFacePipeline(pipeline=pipe)
        else:
            self.client = InferenceClient(token=HF_API_TOKEN, timeout=300)

    def predict_table_name(self, columns: List[str]) -> str:
        prompt_text = """[INST]Given the column names in a database, predict a suitable name for the table that likely represents the data described by these columns.

        Column names: language_id, language_code, language_name
        Table name:[/INST] language

        [Inst]Column names: {columns}
        Table name:[/INST]"""
        joined_columns = ", ".join(columns)

        if USE_LOCAL_MODEL:
            prompt = PromptTemplate(template=prompt_text, input_variables=["columns"])
            llm_chain = LLMChain(prompt=prompt, llm=self.llm)
            args = {"columns": joined_columns}
            return llm_chain.run(args).strip()
        else:
            prompt_text = prompt_text.replace("{columns}", joined_columns)
            answer = self.client.text_generation(
                prompt_text,
                max_new_tokens=self.max_new_tokens,
                model=self.model_name,
            )
            return answer.removeprefix(" ")
