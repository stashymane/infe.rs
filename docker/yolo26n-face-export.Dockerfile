FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python3 \
    python3-pip \
    python3-venv \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxcb1 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
ARG ULTRALYTICS_VERSION=8.4.128
ARG EXECUTORCH_VERSION=1.4.0

RUN python3 -m venv /opt/export-venv \
    && /opt/export-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/export-venv/bin/pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && /opt/export-venv/bin/pip install --no-cache-dir \
        "ultralytics==${ULTRALYTICS_VERSION}" \
        "executorch==${EXECUTORCH_VERSION}" \
        pyyaml

ENV PATH="/opt/export-venv/bin:${PATH}"

COPY yolo26n-face/export.py /usr/local/share/infers/yolo26n-face/export.py
COPY yolo26n-face/entrypoint.sh /usr/local/bin/yolo26n-face-export-entrypoint.sh
RUN chmod +x /usr/local/bin/yolo26n-face-export-entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/yolo26n-face-export-entrypoint.sh"]
