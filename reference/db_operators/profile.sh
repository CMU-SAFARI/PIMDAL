#!/bin/sh

ADVISOR_PATH=~/intel/oneapi/advisor/2025.0/bin64

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/intel/oneapi/2025.0/lib
export OMP_NUM_THREADS=16

cd ~/db_operators/build
${ADVISOR_PATH}/advisor -collect roofline -project-dir selection -- numactl --cpunodebind=0 --membind=0 bin/select
${ADVISOR_PATH}/advisor -collect roofline -project-dir aggregation -- numactl --cpunodebind=0 --membind=0 bin/aggregate
${ADVISOR_PATH}/advisor -collect roofline -project-dir ordering -- numactl --cpunodebind=0 --membind=0 bin/sort
${ADVISOR_PATH}/advisor -collect roofline -project-dir hash_join -- numactl --cpunodebind=0 --membind=0 bin/hash_join
${ADVISOR_PATH}/advisor -collect roofline -project-dir sm_join -- numactl --cpunodebind=0 --membind=0 bin/sm_join