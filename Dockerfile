FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y git

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
