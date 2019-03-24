FROM python:3.7-slim
# FROM ubuntu:latest

COPY /app .
WORKDIR /app
COPY requirements.txt .

RUN apt-get update && \
 apt-get install -y gcc g++ vim && \
 apt-get clean && \
 pip3 install --no-cache-dir --upgrade setuptools && \
 pip3 install --no-cache-dir -r requirements.txt && \
 python3 -m spacy download en_core_web_sm && \
 pip3 list && \
 rm *