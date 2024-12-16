FROM amd64/python:3.8-slim

WORKDIR /usr/app/

RUN pip install -U pip &&\
    pip install mlflow==2.17.2 boto3==1.35.81 minio==7.2.10

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY image_predict.py predict.py
ENTRYPOINT [ "python3", "predict.py", "--run-id" ]
