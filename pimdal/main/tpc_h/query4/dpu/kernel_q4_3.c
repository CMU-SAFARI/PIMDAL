#include <defs.h>
#include <barrier.h>
#include <mram.h>
#include <alloc.h>
#include <stdint.h>
#include <stdio.h>
#include <mutex.h>
#include <mutex_pool.h>
#include <string.h>

#include "datatype.h"
#include "param.h"
#include "hash_join.h"
#include "aggregate.h"
#include "hash_aggr.h"

#define BLOCK_SIZE 64
#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif

__host query_args_t dpu_args;
__host query_res_t dpu_results;

hash_arguments_t hash_args;
part_arguments_t part_args;
merge_arguments_t merge_args;
merge_results_t merge_res;

aggr_results_t aggr_res;

BARRIER_INIT(barrier, NR_TASKLETS);

MUTEX_INIT(mutex);
MUTEX_POOL_INIT(mutexes, 16);

void create_ptr(uint32_t in, uint32_t out, uint32_t count) {

    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptr32* key_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));
    key_ptr32* ptr_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr32)), key_cache, BLOCK_SIZE*sizeof(key_ptr32));

        for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
            key_ptr32 new_ptr = {.key = key_cache[i].key, .ptr = base+i};
            ptr_cache[i] = new_ptr;
        }

        mram_write(ptr_cache, (__mram_ptr void*) (out + base*sizeof(key_ptr32)), BLOCK_SIZE*sizeof(key_ptr32));
    }
}

void load_next(uint32_t in, uint32_t out, uint32_t load, uint32_t count) {
    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptr32* sel_cache = (key_ptr32*) mem_alloc(16*sizeof(key_ptr32));
    key_ptrtext* out_cache = (key_ptrtext*) mem_alloc(16*sizeof(key_ptrtext));

    uint32_t base_tasklet = tasklet_id*16;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*16) {
        uint32_t size_load = base + 16 > count ? count % 16 : 16;

        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr32)), sel_cache, 16*sizeof(key_ptr32));

        for (uint32_t i = 0; i < size_load; i++) {

            // __dma_aligned key_ptr32 read;
            // mram_read((__mram_ptr void*) (key_load + sel_cache[i].ptr*sizeof(key_ptr32)), &read, sizeof(key_ptr32));
            // out_cache[i].key = read.key;

            // Load the next element into the cache
            mram_read((__mram_ptr void*) (load + sel_cache[i].ptr*16), out_cache[i].prio, 16);

            out_cache[i].val = 1;
        }

        mram_write(out_cache, (__mram_ptr void*) (out + base*sizeof(key_ptrtext)), 16*sizeof(key_ptrtext));
    }
}

void out_val(uint32_t in, uint32_t count) {
    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    uint32_t* out_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        uint32_t size_load = base + BLOCK_SIZE > count ? count % BLOCK_SIZE : BLOCK_SIZE;

        mram_read((__mram_ptr void*) (in + base*sizeof(uint32_t)), out_cache, BLOCK_SIZE*sizeof(uint32_t));

        for (uint32_t i = 0; i < size_load; i++) {
            printf("%u\n", out_cache[i]);
        }
    }
}

void out_char(uint32_t in, uint32_t count) {
    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    char* out_cache = (char*) mem_alloc(16*16);

    uint32_t base_tasklet = tasklet_id*16;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*16) {
        uint32_t size_load = base + 16 > count ? count % 16 : 16;

        mram_read((__mram_ptr void*) (in + base*16), out_cache, 16*16);

        for (uint32_t i = 0; i < size_load; i++) {
            printf("%s\n", &out_cache[i*16]);
        }
    }
}

key_ptr_t unique(key_ptr_t curr_val, __attribute__((unused)) key_ptr_t element) {

    return curr_val;
}

int main() {

    uint32_t tasklet_id = me();

    uint32_t size = 524288;
    uint32_t buffer_1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t buffer_2 = (uint32_t) (buffer_1 + size*sizeof(key_ptr32));
    uint32_t buffer_3 = (uint32_t) (buffer_2 + size*sizeof(key_ptr32));
    uint32_t buffer_4 = (uint32_t) (buffer_3 + size*sizeof(key_ptr32));
    uint32_t part = (uint32_t) (buffer_4 + size*sizeof(key_ptr32));
    uint32_t table = (uint32_t) (part + size*sizeof(key_ptr32));
    uint32_t part_sizes = (uint32_t) (table + size*sizeof(key_ptr32));
    
    create_ptr(buffer_1, buffer_4, dpu_args.o_count);

    if (tasklet_id == 0) {
        hash_args.in_ptr = buffer_4;
        hash_args.size = dpu_args.o_count;
        hash_args.shift = 13;
        hash_args.table_ptr = table;
        hash_args.table_size = 8192;
    }
    barrier_wait(&barrier);

    hash_kernel(&hash_args);
    barrier_wait(&barrier);


    if (tasklet_id == 0) {
        part_args.in_ptr = buffer_2;
        part_args.part_ptr = part;
        part_args.shift = 13;
        part_args.part_sizes = part_sizes;
        part_args.size = dpu_args.l_count;
        part_args.part_n = 32;
    }
    barrier_wait(&barrier);

    part_kernel(&part_args);
    barrier_wait(&barrier);

    if (tasklet_id == 0) {
        merge_args.table_ptr = table;
        merge_args.size = dpu_args.l_count;
        merge_args.in_ptr = part;
        merge_args.out_ptr = buffer_4;
        merge_args.part_sizes = part_sizes;
        merge_args.part_n = 32;
    }
    barrier_wait(&barrier);

    merge_kernel(&merge_args, &merge_res);
    barrier_wait(&barrier);

    aggr_arguments_t aggr_args = {.in = buffer_4, .out = buffer_2, .size = merge_res.out_n, .aggr = unique};
    group_kernel(&aggr_args, &aggr_res);

    load_next(buffer_2, buffer_4, buffer_3, aggr_res.t_count);
    barrier_wait(&barrier);

    if (tasklet_id == 0) {
        uint64_t count = aggr_res.t_count;
        mram_write(&count, (__mram_ptr void*) buffer_1, sizeof(uint64_t));
    }
 
    return 0;
}