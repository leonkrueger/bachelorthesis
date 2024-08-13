import os
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from ..large_language_model import LargeLanguageModel


class Llama3Model(LargeLanguageModel):
    def __init__(self) -> None:
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=os.environ["HF_API_TOKEN"]
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=os.environ["HF_API_TOKEN"],
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.loaded_peft_model = False

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )

    def run_prompt(
        self, messages: List[Dict[str, str]], max_new_tokens: int = 30
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        return self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )[0]["generated_text"][len(prompt) :].strip()

    def load_and_set_adapter(self, fine_tuned_model_dir: str = None) -> None:
        if fine_tuned_model_dir:
            if (
                self.loaded_peft_model
                and not fine_tuned_model_dir in self.model.peft_config
            ):
                self.model.add_adapter(fine_tuned_model_dir, fine_tuned_model_dir)
            elif not self.loaded_peft_model:
                self.model = PeftModel.from_pretrained(
                    self.model, fine_tuned_model_dir, fine_tuned_model_dir
                )
                self.loaded_peft_model = True
            self.model.set_adapter(fine_tuned_model_dir)
        elif self.loaded_peft_model:
            # Disable adapters if no adapter should be loaded and some adapter was loaded before
            self.model.disable_adapter()

    def delete_adapter(self, fine_tuned_model_dir: str) -> None:
        self.model.delete_adapter(fine_tuned_model_dir)
