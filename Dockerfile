
FROM python:3.11-slim

WORKDIR /visionguard

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTRAOP_THREADS=2 \
    TF_NUM_INTEROP_THREADS=2 \
    FLASK_DEBUG=false \
    MODEL_PATH=/visionguard/models/mobilenetv2_tuned.keras

COPY requirements.docker.txt .


RUN pip install --no-cache-dir \
    -r requirements.docker.txt \
    && find /usr/local/lib/python3.11/site-packages \
        -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && find /usr/local/lib/python3.11/site-packages \
        -type f -name '*.pyc' -delete


COPY app/ ./app/

RUN mkdir -p /visionguard/models /visionguard/instance

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "app.main:app"]
