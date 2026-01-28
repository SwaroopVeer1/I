# Use CUDA runtime base
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Python
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --upgrade torch diffusers transformers accelerate safetensors xformers runpod Pillow

# Copy worker code
COPY handler.py schemas.py /app/
WORKDIR /app

# Create cache folder for model at runtime
RUN mkdir -p /weights

# Run the serverless worker
CMD ["python", "-u", "handler.py"]
