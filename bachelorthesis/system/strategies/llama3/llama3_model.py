import os
from typing import Dict, List

import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from ..large_language_model import LargeLanguageModel


class Llama3Model(LargeLanguageModel):
    def __init__(self, fine_tuned_model_dir: str = None) -> None:
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=os.environ["HF_API_TOKEN"]
        )

        if fine_tuned_model_dir:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=os.environ["HF_API_TOKEN"],
                quantization_config=bnb_config,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(base_model, fine_tuned_model_dir)
            model = model.merge_and_unload()
        else:
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

    def run_prompt(
        self, messages: List[Dict[str, str]], max_new_tokens: int = 30
    ) -> str:
        prompt = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        terminators = [
            self.pipe.tokenizer.eos_token_id,
            self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        return self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )[0]["generated_text"][len(prompt) :].strip()
