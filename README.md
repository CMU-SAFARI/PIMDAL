# Overview

PIMDAL (PIM Data Analytics Library) is an implementation of DB operators and 5 TPC-H queries on the UPMEM PIM system. Additionally we provide code to generate the TPC-H data and reference implementations on the CPU and GPU.

# PIMDAL structure

The UPMEM PIM code is located in the *pimdal* directory. These are the different components of the library:

* *aggregate*: Hash and sort based aggregation implementation
* *general*: General arithmetic functions
* *hash*: Hash table creation and hash functions
* *join*: Hash and sort-merge join implementation
* *select*: Selection implementation
* *sort*: Sort implementation used for ordering and other operators
* *support*: Functions used for data transfers

The microbenchmarks and the selected TPC-H queries are implemented in the *main* folder.

The *.devcointainer* directory contains files for running the code using Visual Studio Code developer containers. The included Dockerfile can be used to create a container including all the necessary dependencies on a local system.

# Dependencies

The code relies on the *Apache Arrow* library.\
<https://arrow.apache.org/install/>\
The *Apache Arrow* installation needs to have been built with the following capabilities:

    -DARROW_BUILD_STATIC=OFF -DARROW_WITH_RE2=OFF -DARROW_WITH_UTF8PROC=OFF -DARROW_COMPUTE=ON \
    -DARROW_TESTING=ON -DARROW_ACERO=ON -DARROW_DATASET=ON -DARROW_PARQUET=ON -DARROW_WITH_SNAPPY=ON

# UPMEM Execution

The UPMEM code can either be run on a system equiped with UPMEM DIMMs or using the simulator in the UPMEM SDK:\
<http://sdk-releases.upmem.com/2023.2.0/ubuntu_20.04/upmem-2023.2.0-Linux-x86_64.tar.gz>\
The provided docker file creates a container with the UPMEM SDK and all other dependencies installed.
To activate the UPMEM SDK run:
    source /upmem-sdk-2023.2.0/upmem_env.sh

The UPMEM simulator does not provide any performance measurement or estimation capabilities! To measure performance the code has to be run on the physical UPMEM system.

The components depending on apache arrow have to be built using *cmake*. They include a *CMakeLists.txt* in the top directory. To build them run:

    cmake -DCMAKE*BUILD*TYPE=Release -S . -B ./build

Cmake creates a *build* directory containing all build files and inside it also a *bin* directory containing the executables. The executables can be run in the corresponding directories by executing the binary starting with *host*.

All the UPMEM benchmarks are implemented in the *pimdal* directory. The number of DPUs and problem size for the micro benchmarks can be changed in *CMakeLists.txt*, in the top level directory.
For running the TPC-H queries the generated data has to be copied in *Apache Parquet* format to *pimdal/main/tpc_h/data* or in a path specified in *pimdal/main/tpc_h/reader/read_table.cpp*.

# Reference Code

The *reference/pyarrow* directory contains the code used to generate and transform the TPC-H data together with the CPU and GPU reference code.

* *generate.py*: Generates the TPC-H data using DuckDB in python, transforms them to the desired format and stores it in a parquet file.
* *cpu*: *micro* includes the micro-benchmarks and *tpc* the TPC-H reference code on the CPU.
* *gpu*: Similarly, *micro* includes the micro-benchmarks and *tpc* the TPC-H reference code on the GPU.

All the python code has to be executed from the top-level *reference* directory. The converted data is also created in this directory, in the *data* directory.

The *reference/db_operators* includes an implementation of the DB operators from scratch. Its main purpose is to be used with profilers like Intel Advisor. It is built using *cmake*, for example by calling one of the build scripts. It requires the *abseil-cpp* library as a dependency.

Similarly to the UPMEM code, this directory contains a *.devcontainer* file and a *Dockerfile* that installs all dependencies for Visual Studio Code.

# Acknowledgements

We acklowledge support from SAFARI Research Group's industrial partners.