# Base image with CUDA 12.1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11, pip, git, wget
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /.venv
ENV PATH="/.venv/bin:${PATH}"

# Upgrade pip
RUN pip install --upgrade pip

# Install optimized dependencies
RUN pip install \
    torch --extra-index-url https://download.pytorch.org/whl/cu121 \
    diffusers transformers accelerate safetensors \
    xformers==0.0.23 runpod numpy==1.26.3 scipy \
    triton huggingface-hub hf_transfer setuptools Pillow

# Copy code
COPY download_weights.py schemas.py handler.py test_input.json /app/
WORKDIR /app

# Download SDXL Turbo weights to /weights
RUN mkdir -p /weights
RUN python download_weights.py

# Environment variables
ENV PYTHONUNBUFFERED=1

# Start the handler
CMD ["python", "-u", "handler.py"]
