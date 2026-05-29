FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-downloaded packages and install without internet
COPY pip_cache/ /pip_cache/
COPY requirements.txt .

RUN pip install --no-cache-dir --no-index \
    --find-links=/pip_cache \
    -r requirements.txt

# Copy application code
COPY . .

RUN mkdir -p /data

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
