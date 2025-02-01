/*
* Author: Manos Frouzakis
* Date: 11.11.2022
* Full quicksort in MRAM using a distributed workload sharing method.
*/

#include <defs.h>
#include <barrier.h>
#include <mram.h>
#include <alloc.h>
#include <stdint.h>
#include <stdio.h>
#include <mutex.h>

#include "sort_func.c"
#include "sort.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif
#ifndef NR_SPLITS
#define NR_SPLITS 512
#endif

// Store the offsets for each block split through quicksort
__mram uint64_t indices_loc[NR_SPLITS][NR_TASKLETS];
// Store prefix sum for offsets of a particular split over all blocks
__mram uint64_t indices_off[NR_SPLITS][NR_TASKLETS];

uint32_t split_sel = 0;

extern barrier_t barrier;

MUTEX_INIT(mutex);

int sort_kernel(sort_arguments_t *input_args) {
    uint32_t tasklet_id = me();
    uint32_t nr_el = input_args->nr_elements;
    uint32_t nr_el_tasklets = tasklet_id < nr_el % NR_TASKLETS ? nr_el/NR_TASKLETS + 1 : nr_el/NR_TASKLETS;
    uint32_t base_tasklet;
    if (tasklet_id < nr_el % NR_TASKLETS) {
        base_tasklet = tasklet_id * (nr_el/NR_TASKLETS + 1);
    }
    else {
        base_tasklet = tasklet_id * (nr_el/NR_TASKLETS) + nr_el % NR_TASKLETS;
    }

    T* buffer_addr = (T*) input_args->in;
    T* sorted = (T*) input_args->out;

    if (tasklet_id == 0) {
        mem_reset();
    }
    barrier_wait(&barrier);
    T* local_cache = (T*) mem_alloc(2*BLOCK_SIZE*sizeof(T));

    // Determine number of elements in each partition of quicksort
    uint32_t pivot_prev = input_args->pivot;
    for (uint32_t i = NR_SPLITS/2; i > 0; i >>= 1) {
        uint64_t pivot = pivot_prev >> 1;
        for (uint32_t split = i; split < NR_SPLITS; split += i*2) {
            uint64_t start = 0;
            if (split > i) {
                mram_read((__mram_ptr void*) &indices_loc[split-1-i][tasklet_id], &start, sizeof(uint64_t));
            }
            uint64_t nr_elements_split;
            if (split+i < NR_SPLITS) {
                mram_read((__mram_ptr void*) &indices_loc[split-1+i][tasklet_id], &nr_elements_split, sizeof(uint64_t));
                nr_elements_split -= start;
            }
            else {
                nr_elements_split = nr_el_tasklets - start;
            }
            uint64_t index_tmp = sort_blocks(buffer_addr+base_tasklet+start, buffer_addr+base_tasklet+start,
                                             nr_elements_split, local_cache, local_cache+BLOCK_SIZE, pivot);
            index_tmp += start;
            mram_write(&index_tmp, (__mram_ptr void*) &indices_loc[split-1][tasklet_id], sizeof(uint64_t));
            pivot += pivot_prev;
        }
        pivot_prev >>= 1;
    }

    // Calculate the partition size from the offsets
    indices_loc[NR_SPLITS-1][tasklet_id] = nr_el_tasklets - indices_loc[NR_SPLITS-2][tasklet_id];
    for (int32_t i = NR_SPLITS-3; i >= 0; i--) {
        indices_loc[i+1][tasklet_id] = indices_loc[i+1][tasklet_id] - indices_loc[i][tasklet_id];
    }

   barrier_wait(&barrier);

    // Calculate the offsets in the output array
    for (uint32_t split = tasklet_id; split < NR_SPLITS; split += NR_TASKLETS) {
        indices_off[split][0] = indices_loc[split][0];
        for (uint32_t i = 1; i < NR_TASKLETS; i++) {
            indices_off[split][i] = indices_off[split][i-1] + indices_loc[split][i];
        }
    }

    barrier_wait(&barrier);

    // Merge partitions into MRAM
    reorder((T*) (buffer_addr+base_tasklet), (T*) sorted, local_cache, tasklet_id,
            (uint64_t (*)[NR_TASKLETS]) indices_loc, (uint64_t (*)[NR_TASKLETS]) indices_off);

    barrier_wait(&barrier);

    // Do the final sort on the partitions
    uint32_t split_task;
    mutex_lock(mutex);
    split_task = split_sel;
    split_sel++;
    mutex_unlock(mutex);
    while (split_task < NR_SPLITS) {
        uint32_t byte_offset = 0;
        for (uint32_t i = 0; i < split_task; i++) {
            byte_offset += indices_off[i][NR_TASKLETS-1];
        }
        nr_el_tasklets = indices_off[split_task][NR_TASKLETS-1];
        if (nr_el_tasklets > 0) {
            sort_full((T*) (sorted+byte_offset), (T*) (sorted+byte_offset), nr_el_tasklets, local_cache, 1);
        }

        mutex_lock(mutex);
        split_task = split_sel;
        split_sel++;
        mutex_unlock(mutex);
    }

    return 0;
}