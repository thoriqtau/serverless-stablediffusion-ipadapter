FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1

# Upgrade apt packages and install required dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
      python3-dev \
      python3-pip \
      python3.10-venv \
      fonts-dejavu-core \
      rsync \
      git \
      git-lfs \
      jq \
      moreutils \
      aria2 \
      wget \
      curl \
      unzip \
      libglib2.0-0 \
      libsm6 \
      libgl1 \
      libxrender1 \
      libxext6 \
      ffmpeg \
      libgoogle-perftools4 \
      libtcmalloc-minimal4 \
      procps && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean -y

# Set the working directory
WORKDIR /workspace

RUN git clone https://github.com/thoriqtau/serverless-stablediffusion-ipadapter.git

# Install the worker dependencies
WORKDIR /workspace/serverless-stablediffusion-ipadapter/src
RUN pip3 install --no-cache-dir torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir runpod && \
    pip3 install -r requirements.txt

# Download the checkpoints
RUN python3 download_checkpoints.py

# Download insightface checkpoints
RUN cd /workspace/serverless-stablediffusion-ipadapter/src && \
    mkdir -p insightface_models/models && \
    cd insightface_models/models && \
    wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    mkdir buffalo_l && \
    cd buffalo_l && \
    unzip ../buffalo_l.zip

# Docker container start script
COPY --chmod=755 start_standalone.sh /start.sh

# Start the container
ENTRYPOINT /start.sh

