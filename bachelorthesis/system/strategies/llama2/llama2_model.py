import re
from enum import Enum

from huggingface_hub import InferenceClient
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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

        if model_type == LLama2ModelType.NON_FINE_TUNED_API:
            self.client = InferenceClient(token=huggingface_api_token, timeout=300)
        else:
            if model_type == LLama2ModelType.NON_FINE_TUNED_LOCAL:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, device_map="auto"
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_dir)
                model = AutoModelForCausalLM.from_pretrained(
                    fine_tuned_model_dir, load_in_4bit=True, device_map="auto"
                )

            # tokenizer.pad_token = tokenizer.unk_token
            # model.config.pad_token_id = tokenizer.pad_token_id
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

    def run_prompt(self, prompt_text: str, max_tokens: int) -> str:
        """Runs a prompt on a Llama2 model and returns its answer"""
        if self.model_type == LLama2ModelType.NON_FINE_TUNED_API:
            return self.client.text_generation(
                prompt_text,
                max_new_tokens=max_tokens,
                model=self.model_name,
            ).strip()
        else:
            prompt = PromptTemplate(template=prompt_text)
            llm_chain = LLMChain(prompt=prompt, llm=self.llm)
            return llm_chain.run().strip()

    def predict_table_name(self, query_data: QueryData) -> str:
        database_string = (
            "\n".join(
                [
                    f"- Table: {table}, Columns: [{', '.join([column[0] for column in columns])}]"
                    for table, columns in query_data.database_state.items()
                ]
            )
            if len(query_data.database_state) > 0
            else "No table exists yet"
        )

        return re.search(
            r"Table: (?P<table>\S+)",
            self.run_prompt(
                "You are an intelligent database that predicts on which table a SQL-insert should be executed. "
                "The inserts can contain abbreviated or synonymous names. The table and column names can be missing entirely. "
                "You should then predict your result based on the available information. "
                "You give the output in the form 'Table: {table_name}'. If there is a suitable table in the database, "
                "you replace '{table_name}' with its name. Else, you replace '{table_name}' with a suitable name for a database table. "
                "You don't give any explanation for your result.",
                "Predict the table for this example:\n"
                f"Query: {query_data.query}\n"
                f"Database State:\n{database_string}",
                10,
            ),
        ).group("table")
