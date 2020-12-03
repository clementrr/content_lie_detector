FROM python:3.8.6-buster

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY api /api
COPY content_lie_detector /content_lie_detector

CMD uvicorn api.fast:app --reload --host 0.0.0.0 --port $PORT