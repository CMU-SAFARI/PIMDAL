#include <defs.h>
#include <barrier.h>
#include <mram.h>
#include <alloc.h>
#include <stdio.h>
#include <mutex.h>
#include <perfcounter.h>

#include "join.h"
#include "datatype.h"
#include "sort.h"
#include "sort_merge.h"

#define BLOCK_SIZE 64
#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif

#ifndef INNER_SIZE
#define INNER_SIZE 100000
#endif
#ifndef OUTER_SIZE
#define OUTER_SIZE 1000000
#endif

__host join_arguments_t join_args;
__host join_results_t join_res;
__host uint64_t cycles_1;
__host uint64_t cycles_2;
__host uint64_t cycles_3;

sort_arguments_t sort_args;
merge_arguments_t merge_args;
merge_results_t merge_res;

BARRIER_INIT(barrier, NR_TASKLETS);

MUTEX_INIT(mutex);

void create_ptr(uint32_t *in, key_ptr32 *out, uint32_t start) {
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
        key_ptr32 new_ptr = {.key = in[i], .ptr = start+i};
        out[i] = new_ptr;
    }
}

void out_val(key_ptr32 *in, uint32_t *out) {
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
        out[i] = in[i].key;
    }
}

extern int main_kernel1(void);
extern int main_kernel2(void);

int (*kernels[2])(void) = {main_kernel1, main_kernel2};

int main(void) {
    return kernels[join_args.kernel_sel]();
}

int main_kernel1() {

    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    barrier_wait(&barrier);

    uint32_t inner_out = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t outer_out = inner_out + INNER_SIZE*sizeof(key_ptr32);
    uint32_t inner_in = outer_out + OUTER_SIZE*sizeof(key_ptr32);
    uint32_t outer_in = inner_in + INNER_SIZE*sizeof(key_ptr32);
    uint32_t inner_indices = outer_in + OUTER_SIZE*sizeof(key_ptr32);
    uint32_t outer_indices = inner_indices + join_args.nr_splits*sizeof(uint64_t);

    uint32_t* val_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));
    key_ptr32* ptr_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < INNER_SIZE; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (inner_out + base*sizeof(uint32_t)), val_cache, BLOCK_SIZE*sizeof(uint32_t));

        create_ptr(val_cache, ptr_cache, join_args.ptr_inner+base);

        mram_write(ptr_cache, (__mram_ptr void*) (inner_in + base*sizeof(key_ptr32)), BLOCK_SIZE*sizeof(key_ptr32));
    }
    barrier_wait(&barrier);

    key_ptr_t pivot = {.key = join_args.range};
    key_ptr_t start_val = {.key = 0};
    if (tasklet_id == 0) {
        sort_args.in = inner_in;
        sort_args.nr_elements = INNER_SIZE;
        sort_args.out = inner_out;
        sort_args.indices = inner_indices;
        sort_args.nr_splits = join_args.nr_splits;
        sort_args.pivot = pivot;
        sort_args.start = start_val;
    }
    barrier_wait(&barrier);

#if PERF == 1
    if (tasklet_id == 0) {
        perfcounter_config(COUNT_CYCLES, true);
    }
    barrier_wait(&barrier);
#elif PERF == 2
    if (tasklet_id == 0) {
        perfcounter_config(COUNT_INSTRUCTIONS, true);
    }
    barrier_wait(&barrier);
#endif

    sort_part_kernel(&sort_args);

#if PERF > 0
    if (tasklet_id == 0) {
        cycles_1 = perfcounter_get();
    }
    barrier_wait(&barrier);
#endif

    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < OUTER_SIZE; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (outer_out + base*sizeof(uint32_t)), val_cache, BLOCK_SIZE*sizeof(uint32_t));

        create_ptr(val_cache, ptr_cache, join_args.ptr_outer+base);

        mram_write(ptr_cache, (__mram_ptr void*) (outer_in + base*sizeof(key_ptr32)), BLOCK_SIZE*sizeof(key_ptr32));
    }
    barrier_wait(&barrier);

    if (tasklet_id == 0) {
        sort_args.in = outer_in;
        sort_args.nr_elements = OUTER_SIZE;
        sort_args.out = outer_out;
        sort_args.indices = outer_indices;
        sort_args.nr_splits = join_args.nr_splits;
        sort_args.pivot = pivot;
        sort_args.start = start_val;
    }
    barrier_wait(&barrier);

#if PERF == 1
    if (tasklet_id == 0) {
        perfcounter_config(COUNT_CYCLES, true);
    }
    barrier_wait(&barrier);
#elif PERF == 2
    if (tasklet_id == 0) {
        perfcounter_config(COUNT_INSTRUCTIONS, true);
    }
    barrier_wait(&barrier);
#endif

    sort_part_kernel(&sort_args);

#if PERF > 0
    if (tasklet_id == 0) {
        cycles_2 = perfcounter_get();
    }
    barrier_wait(&barrier);
#endif

    return 0;
}

int main_kernel2() {

    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    barrier_wait(&barrier);

    uint32_t inner_in = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t outer_in = inner_in + join_args.offset_outer*sizeof(key_ptr32);
    uint32_t inner_out = outer_in + join_args.n_el_outer*sizeof(key_ptr32);
    uint32_t outer_out = inner_out + join_args.n_el_inner*sizeof(key_ptr32);

    key_ptr_t pivot = {.key = join_args.range};
    key_ptr_t start_val = {.key = join_args.start};
    if (tasklet_id == 0) {
        sort_args.in = inner_in;
        sort_args.nr_elements = join_args.n_el_inner;
        sort_args.out = inner_out;
        sort_args.nr_splits = 64;
        sort_args.pivot = pivot;
        sort_args.start = start_val;
    }
    barrier_wait(&barrier);

#if PERF == 1
    if (tasklet_id == 0) {
        perfcounter_config(COUNT_CYCLES, true);
    }
    barrier_wait(&barrier);
#elif PERF == 2
    if (tasklet_id == 0) {
        perfcounter_config(COUNT_INSTRUCTIONS, true);
    }
    barrier_wait(&barrier);
#endif

    sort_kernel(&sort_args);
    barrier_wait(&barrier);

#if PERF > 0
    if (tasklet_id == 0) {
        cycles_1 = perfcounter_get();
    }
    barrier_wait(&barrier);
#endif

    if (tasklet_id == 0) {
        sort_args.in = outer_in;
        sort_args.nr_elements = join_args.n_el_outer;
        sort_args.out = outer_out;
        sort_args.nr_splits = 64;
        sort_args.pivot = pivot;
        sort_args.start = start_val;
    }
    barrier_wait(&barrier);

#if PERF == 1
    if (tasklet_id == 0) {
        perfcounter_config(COUNT_CYCLES, true);
    }
    barrier_wait(&barrier);
#elif PERF == 2
    if (tasklet_id == 0) {
        perfcounter_config(COUNT_INSTRUCTIONS, true);
    }
    barrier_wait(&barrier);
#endif

    sort_kernel(&sort_args);
    barrier_wait(&barrier);

#if PERF > 0
    if (tasklet_id == 0) {
        cycles_2 = perfcounter_get();
    }
    barrier_wait(&barrier);
#endif

    if (tasklet_id == 0) {
        merge_args.inner = inner_out;
        merge_args.outer = outer_out;
        merge_args.size_inner = join_args.n_el_inner;
        merge_args.size_outer = join_args.n_el_outer;
        merge_args.out = inner_in;
    }
    barrier_wait(&barrier);

#if PERF == 1
    if (tasklet_id == 0) {
        perfcounter_config(COUNT_CYCLES, true);
    }
    barrier_wait(&barrier);
#elif PERF == 2
    if (tasklet_id == 0) {
        perfcounter_config(COUNT_INSTRUCTIONS, true);
    }
    barrier_wait(&barrier);
#endif

    merge_kernel(&merge_args, &merge_res);
    barrier_wait(&barrier);

#if PERF > 0
    if (tasklet_id == 0) {
        cycles_3 = perfcounter_get();
    }
    barrier_wait(&barrier);
#endif

    if (tasklet_id == 0) {
        join_res.count = merge_res.t_count;
    }

    return 0;
}