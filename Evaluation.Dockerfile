FROM nvidia/cuda:11.0.3-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt install -y git libsndfile1-dev python3 python3-pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

COPY requirements.txt ./
RUN pip install -r requirements.txt
CMD ["python3", "evaluation.py"]