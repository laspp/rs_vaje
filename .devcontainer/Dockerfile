FROM ubuntu:latest

# GEM5
RUN apt update \
    && apt install -y git build-essential m4 scons python3-dev zlib1g zlib1g-dev libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev python3 python-is-python3 python3-pydot
RUN git clone https://github.com/gem5/gem5
RUN printf 'RUBY=y\nUSE_MULTIPLE_PROTOCOLS=y\nPROTOCOL="MULTIPLE"\nRUBY_PROTOCOL_MOESI_AMD_Base=y\nRUBY_PROTOCOL_MESI_Two_Level=y\nRUBY_PROTOCOL_MESI_Three_Level=y\nRUBY_PROTOCOL_MESI_Three_Level_HTM=y\nRUBY_PROTOCOL_MI_example=y\nRUBY_PROTOCOL_MOESI_CMP_directory=y\nRUBY_PROTOCOL_MOESI_CMP_token=y\nRUBY_PROTOCOL_MOESI_hammer=y\nRUBY_PROTOCOL_Garnet_standalone=y\nRUBY_PROTOCOL_CHI=y\nRUBY_PROTOCOL_MSI=y\nBUILD_ISA=y\nUSE_RISCV_ISA=y' > /gem5/build_opts/RISCV
RUN cd gem5 && scons build/RISCV/gem5.opt -j $(nproc)

ENV GEM5_ROOT=/gem5
ENV GEM_PATH=/gem5/build/RISCV

# RISCV compiler
RUN apt install -y gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
# aliases for GEM5
RUN ln -s /usr/bin/riscv64-linux-gnu-gcc /usr/bin/riscv64-unknown-linux-gnu-gcc
RUN ln -s /usr/bin/riscv64-linux-gnu-g++ /usr/bin/riscv64-unknown-linux-gnu-g++

# build M5
RUN cd /gem5/util/m5 && scons riscv.CROSS_COMPILE=riscv64-linux-gnu- build/riscv/out/m5

# gem5 alias
RUN ln -s /gem5/build/RISCV/gem5.opt /usr/bin/gem5

# more compatibility stuff (original scripts are made for hpc)
# so we made some links and dummy passthrough commands to keep them working

RUN ls - s /gem5/build/RISCV /gem5/build/RISCV_ALL_RUBY

RUN mkdir -p /d/hpc/projects/FRI/GEM5/gem5_workspace && ln -s /gem5 /d/hpc/projects/FRI/GEM5/gem5_workspace

RUN printf '#!/usr/bin/env bash\nshift 2 # ignore "apptainer exec $GEM5_WORKSPACE/gem5_rv.sif"\n"$@"\n' > /usr/bin/apptainer && chmod +x /usr/bin/apptainer

RUN printf '#!/usr/bin/env bash\n"$@"\n' > /usr/bin/srun && chmod +x /usr/bin/srun

