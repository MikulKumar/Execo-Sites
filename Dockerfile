FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        libffi-dev \
        wget \
        gnupg \
    && rm -rf /var/lib/apt/lists/*

# Copy your app files
COPY . /app

# Upgrade pip & install all Python packages at once
RUN pip install --upgrade pip setuptools wheel
RUN pip install numpy==1.25.2 faiss-cpu==1.12.0 openai flask gunicorn

# Expose port
EXPOSE 10000

# Start app
CMD ["gunicorn", "proto-1:app", "-b", "0.0.0.0:10000", "--workers", "2"]
