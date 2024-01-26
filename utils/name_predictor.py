from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class NamePredictor:
    def __init__(self) -> None:
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

        tokenizer.pad_token = tokenizer.unk_token
        model.config.pad_token_id = tokenizer.pad_token_id
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            do_sample=True,
            max_new_tokens=1,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def predict_table_name(self, columns: list[str]) -> str:
        promt_text = """[INST]Given the column names in a database, predict a suitable name for the table that likely represents the data described by these columns.

        Column names: language_id, language_code, language_name
        Table name:[/INST] language

        [Inst]Column names: {columns}
        Table name:[/INST]"""
        prompt = PromptTemplate(template=promt_text, input_variables=["columns"])
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)

        args = {"columns": ", ".join(columns)}
        answer = llm_chain.batch(args)
        return answer
