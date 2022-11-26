FROM python:3.10

WORKDIR /app

ADD dist/* .

RUN pip install *.whl

ADD iris_classifier.joblib .

CMD uvicorn iris.iris_pred:app --host 0.0.0.0 --port 8000