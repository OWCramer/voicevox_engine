# Use NVIDIA CUDA base image for GPU support
# CUDA 11.8 is generally compatible with the ONNX Runtime versions used by Voicevox
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies
# libsndfile1 is required for audio processing
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libsndfile1 \
    curl \
    wget \
    unzip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency definitions
COPY pyproject.toml uv.lock requirements.txt ./

# Install Python dependencies
# We use system python here to simplify the GPU container setup
RUN uv pip install --system --no-cache-dir -r requirements.txt
RUN uv pip install --system --no-cache-dir runpod

# Download VOICEVOX Core (GPU Version)
# Version 0.15.0 is used here. Ensure this matches your engine compatibility.
RUN wget https://github.com/VOICEVOX/voicevox_core/releases/download/0.15.0/voicevox_core-linux-x64-gpu-0.15.0.zip -O core.zip && \
    unzip core.zip && \
    mkdir -p voicevox_core && \
    # Move contents from the extracted folder (name changes with version) to /app/voicevox_core \
    mv voicevox_core-linux-x64-gpu-0.15.0/* voicevox_core/ && \
    rm core.zip && \
    rm -rf voicevox_core-linux-x64-gpu-0.15.0

# Copy the application code
COPY . .

# Set Environment Variables for RunPod and Voicevox
# LD_LIBRARY_PATH is critical for loading the .so files from the core directory
ENV LD_LIBRARY_PATH=/app/voicevox_core:$LD_LIBRARY_PATH
# Force engine to use GPU
ENV VV_USE_GPU=1
ENV VV_CPU_NUM_THREADS=4

# Run the handler
CMD ["python", "-u", "handler.py"]