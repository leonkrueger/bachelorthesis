FROM python:3.10
RUN pip3 install mysql-connector-python tabulate
# RUN python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("all-MiniLM-L6-v2")'
COPY . .
CMD ["python3", "main.py"]