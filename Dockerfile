FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    google-perftools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
    && pip3 install --no-cache-dir -r requirements.txt

COPY fclip.py .
COPY start.sh .

# Make start.sh executable
RUN chmod +x /workspace/start.sh

CMD ["/workspace/start.sh"]
