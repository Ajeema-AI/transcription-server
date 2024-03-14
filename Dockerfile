FROM nvcr.io/nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-dev gcc g++ wget python3-pip uvicorn  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./server /app

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8080


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

