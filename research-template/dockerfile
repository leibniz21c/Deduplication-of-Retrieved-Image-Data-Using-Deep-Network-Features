FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# set environment variables
ENV WORKSPACE /workspace
ENV DATASET_PATH /datasets

# set working directory
WORKDIR /workspace

# copy dependencies
COPY requirements.txt /workspace/

# Dependencies and setting
RUN apt update -y && \
    apt-get install -y wget gcc libc6-dev g++ && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    mkdir /datasets