FROM python:3.12-slim-buster

WORKDIR /app

# Install UV
RUN pip install --no-cache-dir uv

# Copy pyproject.toml (UV configuration file)
COPY trainer/pyproject.toml .

# Sync dependencies using UV
RUN uv sync

# Copy the rest of the application
COPY trainer/ train/

ENV PYTHONUNBUFFERED=1

CMD ["python", "train/train.py"]
