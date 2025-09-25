# leverage the renci python base image
#FROM ghcr.io/translatorsri/renci-python-image:3.12.4
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=5

# Create and set working directory
WORKDIR /repo

COPY requirements.txt .

# get the latest code
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --upgrade pip setuptools && \
    pip install -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* ~/.cache

COPY . .

EXPOSE 6380

ENTRYPOINT ["bash", "main.sh"]
