
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


# Create required directories and non-root user.
RUN mkdir -p /visionguard/models /visionguard/instance \
    && groupadd --system visionguard \
    && useradd --system --gid visionguard \
        --home-dir /visionguard visionguard \
    && chown -R visionguard:visionguard \
        /visionguard/instance /visionguard/app

# Run the Flask application without root privileges.
USER visionguard

EXPOSE 5000

# Check whether the application is responding.
HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=90s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=5)" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "app.main:app"]
