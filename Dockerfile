FROM python:3.10
RUN pip install mysql-connector-python tabulate openai
CMD ["python3", "main.py"]