# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QWEN_DATA_DIR=/data

RUN groupadd --system app \
    && useradd --system --gid app --create-home app

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY --chown=app:app . .

RUN mkdir -p /data \
    && chown -R app:app /app /data

USER app

VOLUME ["/data"]
EXPOSE 7860
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/login', timeout=4).read(1)"

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "7860"]
