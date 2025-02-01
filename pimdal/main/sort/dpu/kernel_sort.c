#include <defs.h>
#include <barrier.h>
#include <mram.h>
#include <alloc.h>
#include <stdio.h>
#include <mutex.h>
#include <perfcounter.h>

#include "args.h"
#include "datatype.h"
#include "sort.h"
#include "sort_merge.h"

#define BLOCK_SIZE 64
#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 100000
#endif

__host kernel_arguments_t kernel_args;
__host uint64_t cycles;

sort_arguments_t sort_args;

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
    return kernels[kernel_args.kernel_sel]();
}

int main_kernel1() {

    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    barrier_wait(&barrier);

    uint32_t sort_out = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t sort_in = sort_out + BUFFER_SIZE*sizeof(key_ptr32);
    uint32_t sort_indices = sort_in + BUFFER_SIZE*sizeof(key_ptr32);

    uint32_t* val_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));
    key_ptr32* ptr_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < BUFFER_SIZE; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (sort_out + base*sizeof(uint32_t)), val_cache, BLOCK_SIZE*sizeof(uint32_t));

        create_ptr(val_cache, ptr_cache, base);

        mram_write(ptr_cache, (__mram_ptr void*) (sort_in + base*sizeof(key_ptr32)), BLOCK_SIZE*sizeof(key_ptr32));
    }
    barrier_wait(&barrier);

    key_ptr_t pivot = {.key = kernel_args.range};
    key_ptr_t start_val = {.key = 0};
    if (tasklet_id == 0) {
        sort_args.in = sort_in;
        sort_args.nr_elements = BUFFER_SIZE;
        sort_args.out = sort_out;
        sort_args.indices = sort_indices;
        sort_args.nr_splits = kernel_args.nr_splits;
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
        cycles = perfcounter_get();
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

    uint32_t sort_in = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t sort_out = sort_in + kernel_args.offset_outer*sizeof(key_ptr32);
    uint32_t sort_indices = sort_out + BUFFER_SIZE*sizeof(key_ptr32);

    uint32_t* val_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));
    key_ptr32* ptr_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));

    key_ptr_t pivot = {.key = kernel_args.range};
    key_ptr_t start_val = {.key = kernel_args.start};
    if (tasklet_id == 0) {
        sort_args.in = sort_in;
        sort_args.nr_elements = kernel_args.nr_el;
        sort_args.out = sort_out;
        sort_args.indices = sort_indices;
        sort_args.nr_splits = kernel_args.nr_splits;
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

#if PERF > 0
    if (tasklet_id == 0) {
        cycles = perfcounter_get();
    }
    barrier_wait(&barrier);
#endif

    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    val_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));
    ptr_cache = (key_ptr_t*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < kernel_args.nr_el; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (sort_out + base*sizeof(key_ptr_t)), ptr_cache, BLOCK_SIZE*sizeof(key_ptr_t));

        out_val(ptr_cache, val_cache);

        mram_write(val_cache, (__mram_ptr void*) (sort_in + base*sizeof(uint32_t)), BLOCK_SIZE*sizeof(uint32_t));
    }
 
    return 0;
}