FROM python:3.7-slim
# FROM ubuntu:latest

COPY /app .
COPY requirements.txt .
COPY /spacy/en_core_web_sm-2.1.0.tar.gz .
# COPY SSL-Trust-2018.crt .

RUN apt-get update && \
 apt-get install -y gcc g++ vim && \
 apt-get clean && \
 pip3 install --no-cache-dir --trusted-host pypi.python.org --upgrade setuptools && \
 pip3 install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt && \
 pip3 install --no-cache-dir --trusted-host pypi.python.org en_core_web_sm-2.1.0.tar.gz
 
WORKDIR /app

# --cert SSL-Trust-2018.crt 