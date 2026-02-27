# vllm-i64 :: Docker image
# Integer-first inference for token-routed models
#
# Build:  docker build -t vllm-i64 .
# Run:    docker run --gpus all -p 8000:8000 vllm-i64 serve pacific-prime-chat

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip cmake ninja-build git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /build
COPY . .

# Install PyTorch (CUDA 12.4)
RUN pip install torch --index-url https://download.pytorch.org/whl/cu124

# Build CUDA kernels
RUN mkdir -p build && cd build && \
    cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release && \
    ninja

# Install vllm-i64
RUN pip install -e ".[dev]"

# =========================================================================
# Runtime image
# =========================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Copy installed packages and built kernels
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=builder /build /app

ENV PYTHONPATH=/app
EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm_i64.cli"]
CMD ["serve", "pacific-prime-chat", "--port", "8000"]
