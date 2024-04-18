FROM python:3.10
COPY finetuning.requirements.txt ./
RUN pip install -r finetuning.requirements.txt
CMD ["python3", "fine_tuning.py"]