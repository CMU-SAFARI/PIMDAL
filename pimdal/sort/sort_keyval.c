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
#include <string.h>
#include <stdio.h>
#include <mutex.h>

#include "sort_keyval_func.c"
#include "sort.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif
#ifndef NR_SPLITS
#define NR_SPLITS 2048
#endif

// Store the offsets for each block split through quicksort
__mram_noinit uint64_t indices_loc[NR_SPLITS][NR_TASKLETS];
// Store prefix sum for offsets of a particular split over all blocks
__mram_noinit uint64_t indices_off[NR_SPLITS][NR_TASKLETS];

uint32_t split_sel = 0;

extern barrier_t barrier;

extern mutex_id_t mutex;

void partition(uint32_t tasklet_id, key_ptr_t* local_cache, sort_arguments_t *input_args) {
    uint32_t nr_el = input_args->nr_elements;
    uint32_t nr_el_tasklets = tasklet_id < nr_el % NR_TASKLETS ? nr_el/NR_TASKLETS + 1 : nr_el/NR_TASKLETS;
    uint32_t base_tasklet;
    if (tasklet_id < nr_el % NR_TASKLETS) {
        base_tasklet = tasklet_id * (nr_el/NR_TASKLETS + 1);
    }
    else {
        base_tasklet = tasklet_id * (nr_el/NR_TASKLETS) + nr_el % NR_TASKLETS;
    }

    key_ptr_t* buffer_addr = (key_ptr_t*) input_args->in;
    key_ptr_t* sorted = (key_ptr_t*) input_args->out;

    if (tasklet_id == 0) {
        memset((__mram_ptr void*) indices_loc, 0, NR_SPLITS*NR_TASKLETS*sizeof(uint64_t));
    }
    barrier_wait(&barrier);

    //printf("%u nr elements: %d\n", tasklet_id, nr_el_tasklets);
    // Determine number of elements in each partition of quicksort
    uint32_t nr_splits = input_args->nr_splits;
    key_ptr_t pivot_prev = input_args->pivot;
    key_ptr_t start_val = input_args->start;
    for (uint32_t i = nr_splits/2; i > 0; i >>= 1) {
        key_ptr_t pivot = {.key = pivot_prev.key >> 1};
        for (uint32_t split = i; split < nr_splits; split += i*2) {
            int64_t start = 0;
            if (split > i) {
                mram_read((__mram_ptr void*) &indices_loc[split-1-i][tasklet_id], &start, sizeof(uint64_t));
            }
            uint64_t nr_elements_split;
            if (split+i < nr_splits) {
                mram_read((__mram_ptr void*) &indices_loc[split-1+i][tasklet_id], &nr_elements_split, sizeof(uint64_t));
                nr_elements_split -= start;
            }
            else {
                nr_elements_split = nr_el_tasklets - start;
            }
            key_ptr_t pivot_curr = {.key = start_val.key + pivot.key};
            //printf("id: %d index: %d pivot: %lld, start: %lld, nr: %lu\n", tasklet_id, split-1, pivot_curr.key, start, nr_elements_split);
            uint64_t index_tmp = sort_blocks(buffer_addr+base_tasklet+start, buffer_addr+base_tasklet+start,
                                             nr_elements_split, local_cache, local_cache+SORT_BLOCK_SIZE, pivot_curr, 1);
            index_tmp += start;
            mram_write(&index_tmp, (__mram_ptr void*) &indices_loc[split-1][tasklet_id], sizeof(uint64_t));
            // Select the pivot value in the next subarray
            pivot.key += pivot_prev.key;
        }
        // The pivot value is halved to be in the middle of the subarray
        pivot_prev.key >>= 1;
    }

    // Calculate the partition size from the offsets
    indices_loc[nr_splits-1][tasklet_id] = nr_el_tasklets - indices_loc[nr_splits-2][tasklet_id];
    for (int32_t i = nr_splits-3; i >= 0; i--) {
        indices_loc[i+1][tasklet_id] = indices_loc[i+1][tasklet_id] - indices_loc[i][tasklet_id];
    }

   barrier_wait(&barrier);

    // Calculate the offsets in the output array
    for (uint32_t split = tasklet_id; split < nr_splits; split += NR_TASKLETS) {
        indices_off[split][0] = indices_loc[split][0];
        for (uint32_t i = 1; i < NR_TASKLETS; i++) {
            indices_off[split][i] = indices_off[split][i-1] + indices_loc[split][i];
        }
    }

    barrier_wait(&barrier);

    // Merge partitions into MRAM
    reorder((key_ptr_t*) (buffer_addr+base_tasklet), (key_ptr_t*) sorted, local_cache, tasklet_id,
            (uint64_t (*)[NR_TASKLETS]) indices_loc, (uint64_t (*)[NR_TASKLETS]) indices_off, nr_splits);

    barrier_wait(&barrier);
}

int sort_part_kernel(sort_arguments_t *input_args) {

    uint32_t tasklet_id = me();

    if (tasklet_id == 0) {
        mem_reset();
    }
    barrier_wait(&barrier);
    key_ptr_t* local_cache = (key_ptr_t*) mem_alloc(2*SORT_BLOCK_SIZE*sizeof(key_ptr_t));

    partition(tasklet_id, local_cache, input_args);

    // Calculate the offset of partitions in the merged data
    if (input_args->indices != 0) {
        uint64_t* indices_glob = (uint64_t*) input_args->indices;

        for (uint32_t split = tasklet_id; split < input_args->nr_splits; split += NR_TASKLETS) {
            uint64_t elements_tot = 0;
            for (uint32_t i = 0; i < NR_TASKLETS; i++) {
                elements_tot += indices_loc[split][i];
            }
            mram_write(&elements_tot, (__mram_ptr void*) (indices_glob+split), sizeof(uint64_t));
        }
    }

    return 0;
}

int sort_kernel(sort_arguments_t *input_args) {

    uint32_t tasklet_id = me();
    key_ptr_t* sorted = (key_ptr_t*) input_args->out;
    if (tasklet_id == 0) {
        mem_reset();
        split_sel = 0;
    }
    barrier_wait(&barrier);
    key_ptr_t* local_cache = (key_ptr_t*) mem_alloc(2*SORT_BLOCK_SIZE*sizeof(key_ptr_t));

    partition(tasklet_id, local_cache, input_args);

    // Do the final sort on the partitions
    uint32_t split_task;
    mutex_lock(mutex);
    split_task = split_sel;
    split_sel++;
    mutex_unlock(mutex);
    while (split_task < input_args->nr_splits) {
        uint32_t byte_offset = 0;
        for (uint32_t i = 0; i < split_task; i++) {
            byte_offset += indices_off[i][NR_TASKLETS-1];
        }
        uint32_t nr_el_tasklets = indices_off[split_task][NR_TASKLETS-1];
        if (nr_el_tasklets > 0) {
            //printf("%u nr elements: %u\n", tasklet_id, nr_el_tasklets);
            sort_full((key_ptr_t*) (sorted+byte_offset), (key_ptr_t*) (sorted+byte_offset), nr_el_tasklets, local_cache);
        }

        mutex_lock(mutex);
        split_task = split_sel;
        split_sel++;
        mutex_unlock(mutex);
    }

    return 0;
}