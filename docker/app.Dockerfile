# syntax = docker/dockerfile:experimental
FROM python:3.10.13-bookworm

RUN apt update && apt install -y libcairo2-dev libgl1 libgl1-mesa-dev libosmesa6-dev

# Install python environment
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

#ADD https://download.pytorch.org/models/resnet34-b627a593.pth \
#    /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth

# Setup bashrc
COPY ./docker/bashrc /root/.bashrc

# Setup PYTHONPATH
ENV PYTHONPATH=.

# # Set up working directory
WORKDIR /app
