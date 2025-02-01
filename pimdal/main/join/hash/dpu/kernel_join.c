#include <defs.h>
#include <barrier.h>
#include <mram.h>
#include <alloc.h>
#include <stdio.h>
#include <mutex.h>
#include <mutex_pool.h>
#include <perfcounter.h>

#include "datatype.h"
#include "join.h"
#include "hash_join.h"

#define BLOCK_SIZE 256
#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif
#ifndef NR_DPU
#define NR_DPU 4
#endif

#ifndef INNER_SIZE
#define INNER_SIZE 100000
#endif
#ifndef OUTER_SIZE
#define OUTER_SIZE 1000000
#endif
#define NR_PART 8192
#define FILTER 512
#define TABLE_SIZE (NR_PART*256)

hash_arguments_t hash_args;
filter_arguments_t filter_args;
part_arguments_t part_args;
match_arguments_t match_args;
merge_arguments_t merge_args;
merge_results_t merge_res;

__host join_arguments_t join_args;
__host join_results_t join_res;
__host uint64_t cycles_1;
__host uint64_t cycles_2;
__host uint64_t cycles_3;

BARRIER_INIT(barrier, NR_TASKLETS);

MUTEX_INIT(mutex);
MUTEX_POOL_INIT(mutexes, 16);

void create_ptr(uint32_t *in, key_ptr32 *out, uint32_t start) {
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
        key_ptr32 new_ptr = {.key = in[i], .ptr = start+i};
        out[i] = new_ptr;
    }
}

void out_val(key_ptr32 *in, uint32_t *outer, uint32_t *inner) {
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
        outer[i] = in[i].key;
        inner[i] = in[i].ptr;
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

    uint32_t inner_part = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t outer_part = inner_part + INNER_SIZE*sizeof(key_ptr32);
    uint32_t inner_size = outer_part + OUTER_SIZE*sizeof(key_ptr32);
    uint32_t outer_size = inner_size + (NR_DPU+1)*sizeof(uint64_t);
    uint32_t buffer = outer_size + (NR_DPU+1)*sizeof(uint64_t);

    uint32_t* val_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));
    key_ptr32* ptr_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < INNER_SIZE; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (inner_part + base*sizeof(uint32_t)), val_cache, BLOCK_SIZE*sizeof(uint32_t));

        create_ptr(val_cache, ptr_cache, join_args.ptr_inner+base);

        mram_write(ptr_cache, (__mram_ptr void*) (buffer + base*sizeof(key_ptr32)), BLOCK_SIZE*sizeof(key_ptr32));
    }
    barrier_wait(&barrier);

    if (tasklet_id == 0) {
        part_args.in_ptr = buffer;
        part_args.size = INNER_SIZE;
        part_args.shift = 27;
        part_args.part_ptr = inner_part;
        part_args.part_sizes = inner_size;
        part_args.part_n = NR_DPU;
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

    part_kernel(&part_args);
    barrier_wait(&barrier);

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

    val_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));
    ptr_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));

    base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < OUTER_SIZE; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (outer_part + base*sizeof(uint32_t)), val_cache, BLOCK_SIZE*sizeof(uint32_t));

        create_ptr(val_cache, ptr_cache, join_args.ptr_outer+base);

        mram_write(ptr_cache, (__mram_ptr void*) (buffer + base*sizeof(key_ptr32)), BLOCK_SIZE*sizeof(key_ptr32));
    }

    if (tasklet_id == 0) {
        part_args.in_ptr = buffer;
        part_args.size = OUTER_SIZE;
        part_args.shift = 27;
        part_args.part_ptr = outer_part;
        part_args.part_sizes = outer_size;
        part_args.part_n = NR_DPU;
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

    part_kernel(&part_args);
    barrier_wait(&barrier);

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

    uint32_t outer_in = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t inner_in = outer_in + join_args.offset_inner*sizeof(key_ptr32);
    uint32_t table = inner_in + TABLE_SIZE*sizeof(key_ptr32);
    uint32_t part_sizes = table + TABLE_SIZE*sizeof(key_ptr32);

    if (tasklet_id == 0) {
        hash_args.in_ptr = inner_in;
        hash_args.size = join_args.n_el_inner;
        hash_args.shift = 13;
        hash_args.table_ptr = table;
        hash_args.table_size = TABLE_SIZE;
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

    hash_kernel(&hash_args);
    barrier_wait(&barrier);

#if PERF > 0
    if (tasklet_id == 0) {
        cycles_1 = perfcounter_get();
    }
    barrier_wait(&barrier);
#endif

    if (tasklet_id == 0) {
        part_args.in_ptr = outer_in;
        part_args.part_ptr = inner_in;
        part_args.shift = 13;
        part_args.part_sizes = part_sizes;
        part_args.size = join_args.n_el_outer;
        part_args.part_n = NR_PART;
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

    part_kernel(&part_args);
    barrier_wait(&barrier);

#if PERF > 0
    if (tasklet_id == 0) {
        cycles_2 = perfcounter_get();
    }
    barrier_wait(&barrier);
#endif

    if (tasklet_id == 0) {
        merge_args.table_ptr = table;
        merge_args.size = join_args.n_el_outer;
        merge_args.in_ptr = inner_in;
        merge_args.out_ptr = outer_in;
        merge_args.part_sizes = part_sizes;
        merge_args.part_n = NR_PART;
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
        join_res.count = merge_res.out_n;
    }
 
    return 0;
}