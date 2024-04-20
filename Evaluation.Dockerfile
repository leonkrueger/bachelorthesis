FROM nvidia/cuda:11.0.3-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

# Install python (uses version 3.8)
RUN apt install -y git libsndfile1-dev python3 python3-pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install requirements for my system
COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN python3 -c 'from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token="YOUR_HF_API_TOKEN"); AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token="YOUR_HF_API_TOKEN")'

# Execute evaluation script
CMD ["python3", "evaluation.py"]