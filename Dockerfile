FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG CUDA_ARCHS

RUN apt-get update

RUN mkdir -p /workspace
WORKDIR /workspace

# ------------------------------
# Install system dependencies and Google Cloud CLI
# ------------------------------
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsparsehash-dev git \
        libopenblas-dev cmake unzip wget curl ca-certificates gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/share/keyrings \
    && curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg \
       | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
       | tee /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*


# ------------------------------
# Python packages
# ------------------------------
RUN pip3 install --upgrade pip

RUN pip3 install \
        torch==2.5.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

RUN pip3 install \
        pybind11==2.11.1 cmake==3.27.0 ninja==1.11.1 \
        numpy==1.23.0 tqdm==4.65.0 packaging==23.1 \
        pandas==1.5.3 matplotlib==3.6.1 scipy==1.11.1 \
        waymo-open-dataset-tf-2-12-0==1.6.5 rootpath==0.1.1

        
# ------------------------------
# Build Engines
# ------------------------------
RUN git clone https://github.com/UofT-EcoSystem/Minuet.git
COPY assets/Minuet_fp16.patch Minuet/Minuet_fp16.patch
RUN cd Minuet \
    && git checkout e413570949ba1cac778c2905870341e18a64fc51 \
    && git apply Minuet_fp16.patch && rm Minuet_fp16.patch \
    && MINUET_ENABLE_CUDA=1 MINUET_CUDA_ARCH_LIST="$CUDA_ARCHS" pip3 install -e . --no-build-isolation

RUN git clone https://github.com/mit-han-lab/torchsparse.git
COPY assets/torchsparse_k_5.patch torchsparse/torchsparse_k_5.patch
RUN cd torchsparse \
    && git checkout 385f5ce8718fcae93540511b7f5832f4e71fd835 \
    && git apply torchsparse_k_5.patch && rm torchsparse_k_5.patch \
    && FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="$CUDA_ARCHS" pip3 install -e . --no-build-isolation --no-deps


COPY source /workspace/Spira

RUN SPIRA_ENABLE_CUDA=1 SPIRA_CUDA_ARCH_LIST="$CUDA_ARCHS" pip3 install -e /workspace/Spira --no-build-isolation

# ------------------------------
# Switch back to non-root for workspace/runtime
# ------------------------------
WORKDIR /workspace/artifacts
ENV PYTHONPATH=.
