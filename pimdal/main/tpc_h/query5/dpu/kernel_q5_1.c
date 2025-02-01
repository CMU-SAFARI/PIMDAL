#include <defs.h>
#include <barrier.h>
#include <mram.h>
#include <alloc.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <mutex.h>
#include <mutex_pool.h>

#include "datatype.h"
#include "param.h"
#include "sel.h"
#include "hash_join.h"

#define BLOCK_SIZE 64
#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif
#ifndef NR_DPU
#define NR_DPU 4
#endif

__host query_args_t dpu_args;
__host query_res_t dpu_results;

sel_results_t sel_results;
part_arguments_t part_args;
hash_arguments_t hash_args;
merge_arguments_t merge_args;
merge_results_t merge_res;

__mram_noinit_keep uint32_t r_regionkey[16];
__mram_noinit_keep uint8_t r_name[16][32];
__mram_noinit_keep uint32_t n_nationkey[32];
__mram_noinit_keep uint32_t n_regionkey[32];
__mram_noinit_keep uint8_t n_name[32][32];
__mram_noinit_keep uint32_t c_custkey[131072];
__mram_noinit_keep uint32_t c_nationkey[131072];

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

    char* key_cache = (char*) mem_alloc(8*32);
    key_ptr_t* ptr_cache = (key_ptr_t*) mem_alloc(8*sizeof(key_ptr_t));

    uint32_t base_tasklet = tasklet_id*8;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*8) {
        mram_read((__mram_ptr void*) (in + base*32), key_cache, 8*32);

        for (uint32_t i = 0; i < 8; i++) {
            memcpy(ptr_cache[i].key, &key_cache[i*32], 32);
            ptr_cache[i].ptr = base+i;
        }

        mram_write(ptr_cache, (__mram_ptr void*) (out + base*sizeof(key_ptr_t)), 8*sizeof(key_ptr_t));
    }
}

void create_ptr32(uint32_t in, uint32_t out, uint32_t count) {

    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    uint32_t* key_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));
    key_ptr32* ptr_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (in + base*sizeof(uint32_t)), key_cache, BLOCK_SIZE*sizeof(uint32_t));

        for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
            ptr_cache[i].key = key_cache[i];
            ptr_cache[i].ptr = base+i;
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

    key_ptr_t* sel_cache = (key_ptr_t*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));
    key_ptr32* out_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        uint32_t size_load = base + BLOCK_SIZE > count ? count % BLOCK_SIZE : BLOCK_SIZE;

        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr_t)), sel_cache, size_load*sizeof(key_ptr_t));

        for (uint32_t i = 0; i < size_load; i++) {
            __dma_aligned uint32_t read[2];
            uint32_t offset = sel_cache[i].ptr & 1;
            uint32_t addr = sel_cache[i].ptr - offset;
            mram_read((__mram_ptr void*) (load + addr*sizeof(uint32_t)), read, 2*sizeof(uint32_t));

            out_cache[i].key = read[offset];
            out_cache[i].ptr = sel_cache[i].ptr;
        }

        mram_write(out_cache, (__mram_ptr void*) (out + base*sizeof(key_ptr32)), size_load*sizeof(key_ptr32));
    }
}

void load_nextptr(uint32_t in, uint32_t out, uint32_t load, uint32_t count) {
    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptr32* sel_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        uint32_t size_load = base + BLOCK_SIZE > count ? count % BLOCK_SIZE : BLOCK_SIZE;

        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr32)), sel_cache, size_load*sizeof(key_ptr32));

        for (uint32_t i = 0; i < size_load; i++) {
            __dma_aligned uint32_t read[2];
            uint32_t offset = sel_cache[i].key & 1;
            uint32_t addr = sel_cache[i].key - offset;
            mram_read((__mram_ptr void*) (load + addr*sizeof(uint32_t)), read, 2*sizeof(uint32_t));

            sel_cache[i].ptr = sel_cache[i].key;
            sel_cache[i].key = read[offset];
        }

        mram_write(sel_cache, (__mram_ptr void*) (out + base*sizeof(key_ptr32)), size_load*sizeof(key_ptr32));
    }
}

void load_out(uint32_t in, uint32_t out, uint32_t load, uint32_t count) {
    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptr32* sel_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));
    uint32_t* out_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        uint32_t size_load = base + BLOCK_SIZE > count ? count % BLOCK_SIZE : BLOCK_SIZE;

        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr32)), sel_cache, size_load*sizeof(key_ptr32));

        for (uint32_t i = 0; i < size_load; i++) {
            __dma_aligned uint32_t read[2];
            uint32_t offset = sel_cache[i].ptr & 1;
            uint32_t addr = sel_cache[i].ptr - offset;
            mram_read((__mram_ptr void*) (load + addr*sizeof(uint32_t)), read, 2*sizeof(uint32_t));

            out_cache[i] = read[offset];
        }

        mram_write(out_cache, (__mram_ptr void*) (out + base*sizeof(uint32_t)), BLOCK_SIZE*sizeof(uint32_t));
    }
}

void out_val(uint32_t in, uint32_t out, uint32_t count) {
    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    uint32_t* ptr_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));
    uint32_t* out_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        uint32_t size_load = base + BLOCK_SIZE > count ? count % BLOCK_SIZE : BLOCK_SIZE;

        mram_read((__mram_ptr void*) (in + base*sizeof(uint32_t)), ptr_cache, BLOCK_SIZE*sizeof(uint32_t));
        for (uint32_t i = 0; i < size_load; i++) {
            out_cache[i] = ptr_cache[i];
        }
        mram_write(out_cache, (__mram_ptr void*) (out + base*sizeof(uint32_t)), BLOCK_SIZE*sizeof(uint32_t));
    }
}

bool pred_r(key_ptr_t element) {
    bool res = (strstr(element.key, dpu_args.r_region) != NULL);
    return res;
}

int main() {

    uint32_t tasklet_id = me();

    uint32_t size = 524288;
    uint32_t buffer_1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t buffer_2 = (uint32_t) (buffer_1 + size*sizeof(key_ptr_t));
    uint32_t buffer_3 = (uint32_t) (buffer_2 + size*sizeof(key_ptr_t));
    uint32_t sizes = (uint32_t) (buffer_3 + size*sizeof(key_ptr_t));

    create_ptr((uint32_t) r_name, buffer_1, dpu_args.r_count);
    barrier_wait(&barrier);

    /*
    * r_name == ASIA
    */
    sel_arguments_t sel_args = {.in = buffer_1, .out = buffer_2, .pred = &pred_r, .size = dpu_args.r_count};
    sel_kernel(&sel_args, &sel_results);
    barrier_wait(&barrier);

    /*
    * JOIN r_regionkey = n_regionkey
    */
    load_next(buffer_2, buffer_1, (uint32_t) r_regionkey, sel_results.t_count);

    if (tasklet_id == 0) {
        hash_args.in_ptr = buffer_1;
        hash_args.size = sel_results.t_count;
        hash_args.shift = 14;
        hash_args.table_ptr = buffer_3;
        hash_args.table_size = 256;
    }
    barrier_wait(&barrier);

    hash_kernel(&hash_args);
    barrier_wait(&barrier);

    create_ptr32((uint32_t) n_regionkey, buffer_1, dpu_args.r_count);

    if (tasklet_id == 0) {
        __dma_aligned uint64_t set_size[2] = {0, dpu_args.n_count};
        mram_write(set_size,(__mram_ptr void*) sizes, 2*sizeof(uint64_t));

        merge_args.table_ptr = buffer_3;
        merge_args.size = dpu_args.n_count;
        merge_args.in_ptr = buffer_1;
        merge_args.out_ptr = buffer_2;
        merge_args.part_sizes = sizes;
        merge_args.part_n = 1;
    }
    barrier_wait(&barrier);

    merge_kernel(&merge_args, &merge_res);
    barrier_wait(&barrier);

    /*
    * JOIN n_nationkey = c_nationkey
    */
    load_nextptr(buffer_2, buffer_1, (uint32_t) n_nationkey, merge_res.out_n);

    if (tasklet_id == 0) {
        hash_args.in_ptr = buffer_1;
        hash_args.size = merge_res.out_n;
        hash_args.shift = 14;
        hash_args.table_ptr = buffer_3;
        hash_args.table_size = 256;
    }
    barrier_wait(&barrier);

    hash_kernel(&hash_args);
    barrier_wait(&barrier);

    create_ptr32((uint32_t) c_nationkey, buffer_1, dpu_args.c_count);

    if (tasklet_id == 0) {
        __dma_aligned uint64_t set_size[2] = {0, dpu_args.c_count};
        mram_write(set_size,(__mram_ptr void*) sizes, 2*sizeof(uint64_t));

        merge_args.table_ptr = buffer_3;
        merge_args.size = dpu_args.c_count;
        merge_args.in_ptr = buffer_1;
        merge_args.out_ptr = buffer_2;
        merge_args.part_sizes = sizes;
        merge_args.part_n = 1;
    }
    barrier_wait(&barrier);

    merge_kernel(&merge_args, &merge_res);
    barrier_wait(&barrier);

    // Partition c_custkey for JOIN
    load_nextptr(buffer_2, buffer_3, (uint32_t) c_custkey, merge_res.out_n);

    if (tasklet_id == 0) {
        part_args.in_ptr = buffer_3;
        part_args.size = merge_res.out_n;
        part_args.shift = 27;
        part_args.part_ptr = buffer_1;
        part_args.part_sizes = sizes;
        part_args.part_n = NR_DPU;
    }
    barrier_wait(&barrier);

    part_kernel(&part_args);
    barrier_wait(&barrier);

    // Load c_nationkey using partitioning
    load_out(buffer_1, buffer_2, (uint32_t) c_nationkey, merge_res.out_n);

    dpu_results.count = merge_res.out_n;

    return 0;
}