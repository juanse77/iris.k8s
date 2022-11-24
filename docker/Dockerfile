FROM python:3.10

WORKDIR /app

ADD dist/* /app

RUN pip install *.whl

ADD model/iris_classifier.joblib /app

CMD uvicorn iris.iris_pred:app --host 0.0.0.0 --port 8000