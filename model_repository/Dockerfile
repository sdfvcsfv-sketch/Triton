FROM nvcr.io/nvidia/tritonserver:25.09-py3

# Update apt metadata, then install packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages (CPU-only PyTorch)
RUN pip3 install --no-cache-dir \
    torch \
    transformers \
    pillow \
    safetensors \
    numpy \
    tritonclient[http]



#       MSYS_NO_PATHCONV=1 docker run -it --rm \
#   -p8000:8000 -p8001:8001 -p8002:8002 \
#   -v "$(pwd):/models" \
#   triton-ocr \
#   tritonserver --model-repository=/models --strict-model-config=0