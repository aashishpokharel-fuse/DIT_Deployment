FROM python:3.7-slim
WORKDIR /app

COPY ./requirements.txt requirements.txt
RUN apt-get update 

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y python3-opencv

COPY . /app

EXPOSE 5000

ENTRYPOINT  python inference.py
