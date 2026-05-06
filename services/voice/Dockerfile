FROM python:3.11-slim

WORKDIR /app

# System deps (single layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ ffmpeg libffi-dev curl git \
    && rm -rf /var/lib/apt/lists/*

# Python deps cache-friendly
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "setuptools<70" wheel && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Copy app last (cache ашиглана)
COPY src ./src

ENV PYTHONUNBUFFERED=1 \
    WHISPER_MODEL=base \
    WHISPER_DEVICE=cuda

EXPOSE 8005

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8005"]