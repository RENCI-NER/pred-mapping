# Build from a github repo
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=5

# Set working directory
WORKDIR /repo

# Install build tools and git
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Always fetch the latest code from main
RUN git clone --branch main https://github.com/RENCI-NER/pred-mapping.git .

# Install Python dependencies
RUN pip install --upgrade pip setuptools && \
    pip install -r requirements.txt

EXPOSE 6380

ENTRYPOINT ["bash", "main.sh"]


# Build with local Files:
## leverage the renci python base image
##FROM ghcr.io/translatorsri/renci-python-image:3.12.4
#FROM python:3.12-slim
#
#ENV DEBIAN_FRONTEND=noninteractive \
#    PIP_NO_CACHE_DIR=1 \
#    PIP_DEFAULT_TIMEOUT=100 \
#    PIP_RETRIES=5
#
## Create and set working directory
#WORKDIR /repo
#
#COPY requirements.txt .
#
## get the latest code
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends build-essential && \
#    pip install --upgrade pip setuptools && \
#    pip install -r requirements.txt && \
#    rm -rf /var/lib/apt/lists/* ~/.cache
#
#COPY . .
#
#EXPOSE 6380
#
#ENTRYPOINT ["bash", "main.sh"]


#run
#docker run --rm \
#  --platform linux/amd64 \
#  -p 6380:6380
#  -e LLM_API_URL=https://healpaca.apps.renci.org/api/generate \
#  -e CHAT_MODEL=HEALpaca-2.0 \
#  -e EMBEDDING_URL=https://healpaca.apps.renci.org/api/embeddings \
#  -e EMBEDDING_MODEL=nomic-embed-text \
#  containers.renci.org/translator-dev/pred-mapping:latest

