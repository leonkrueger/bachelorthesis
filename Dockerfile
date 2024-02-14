FROM python:3.10
WORKDIR /app
RUN pip install mysql-connector-python tabulate torch accelerate langchain transformers
RUN python -c 'from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1"); tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")'
COPY . /app
CMD ["python3", "main.py"]