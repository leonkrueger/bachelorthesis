FROM python:3.10
RUN pip3 install mysql-connector-python tabulate torch accelerate langchain transformers
RUN python -c 'from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1"); tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")'
COPY . .
CMD ["python3", "main.py"]