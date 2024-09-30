ARG BASE_IMAGE
ARG CUDA_ARCH
FROM docker.io/runpod/pytorch:$BASE_IMAGE AS build-000

# all gpus except for h100 and a100
#ENV TORCH_CUDA_ARCH_LIST="Ampere"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir /app
WORKDIR /app

# Install OS dependencies
RUN apt-get update

# Install Python dependencies
RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir huggingface_hub runpod exllamav2

# Yes. That *is* necessary.
RUN git clone --depth 1 https://github.com/turboderp/exllamav2.git
RUN cd exllamav2 && \
    pip install --no-cache-dir -r requirements.txt
RUN rm -rf exllamav2

COPY handler.py /app/handler.py

ENV MODEL_REPO=""
ENV PROMPT_PREFIX=""
ENV PROMPT_SUFFIX=""
ENV HF_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

CMD ["python3", "-u", "/app/handler.py"]
