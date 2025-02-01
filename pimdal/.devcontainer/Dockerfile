FROM mcr.microsoft.com/devcontainers/cpp:0-ubuntu-20.04

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="none"

# Optionally install the cmake for vcpkg
#COPY ./reinstall-cmake.sh /tmp/

#RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
#        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
#    fi \
#    && rm -f /tmp/reinstall-cmake.sh

# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

RUN apt update && \
DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common && \
add-apt-repository ppa:deadsnakes/ppa && \
DEBIAN_FRONTEND=noninteractive apt install -y wget libedit-dev libxml2 python3.8-dev libelf-dev libnuma-dev python3-pip pkg-config python3.7-dev && \
pip3 install six && \
wget http://sdk-releases.upmem.com/2023.2.0/ubuntu_20.04/upmem-2023.2.0-Linux-x86_64.tar.gz && \
tar xf upmem-2023.2.0-Linux-x86_64.tar.gz && \
rm upmem-2023.2.0-Linux-x86_64.tar.gz && \
mv upmem-2023.2.0-Linux-x86_64/ upmem-sdk-2023.2.0

# Update the base image OS and install wget and python
COPY kitware-archive.sh /
RUN bash /kitware-archive.sh

RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y cmake
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y g++-10
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y rapidjson-dev libjemalloc-dev libboost-all-dev ninja-build


# build arrow from source
RUN git clone https://github.com/apache/arrow.git
RUN cd arrow && \
    git submodule init && \
    git submodule update && \
    git checkout apache-arrow-12.0.1 && \
    mkdir cpp/build && \
    cd cpp/build && \
    cmake -GNinja --preset ninja-debug-minimal -DARROW_BUILD_STATIC=OFF -DARROW_WITH_RE2=OFF -DARROW_WITH_UTF8PROC=OFF -DARROW_COMPUTE=ON \
                  -DARROW_TESTING=ON -DARROW_ACERO=ON -DARROW_DATASET=ON -DARROW_PARQUET=ON -DARROW_WITH_SNAPPY=ON .. && \
    cmake --build . && \
    ninja install

CMD bash -c "source /upmem-sdk-2023.1.0/upmem_env.sh simulator && bash"
