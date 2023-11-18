# syntax = docker/dockerfile:experimental
FROM python:3.10.13-bookworm

RUN apt update && apt install -y libcairo2-dev libgl1

# Install python environment
COPY ./requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,mode=0777,target=/root/.cache/pip pip install -r /tmp/requirements.txt

# Setup bashrc
COPY ./docker/bashrc /root/.bashrc

# Setup PYTHONPATH
ENV PYTHONPATH=.

# # Set up working directory
WORKDIR /app
