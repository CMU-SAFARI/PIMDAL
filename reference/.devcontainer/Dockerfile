FROM nvcr.io/nvidia/rapidsai/base:24.12-cuda12.0-py3.12

USER root
RUN DEBIAN_FRONTEND=noninteractive apt update && apt install -y bison cmake git && rm -rf /var/lib/apt/lists/*
RUN conda install -y duckdb

#RUN  echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
#     wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
#         apt update -y && \
#     DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
#         nsight-compute-2023.1.1 && \
#     rm -rf /var/lib/apt/lists/*
