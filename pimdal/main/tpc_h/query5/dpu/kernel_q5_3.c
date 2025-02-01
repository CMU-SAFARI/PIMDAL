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

hash_arguments_t hash_args;
filter_arguments_t filter_args;
part_arguments_t part_args;
match_arguments_t match_args;
merge_arguments_t merge_args;
merge_results_t merge_res;

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

    key_ptr_t* ptr_cache = (key_ptr_t*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr_t)), ptr_cache, BLOCK_SIZE*sizeof(key_ptr_t));

        for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
            ptr_cache[i].ptr = base+i;
        }

        mram_write(ptr_cache, (__mram_ptr void*) (out + base*sizeof(key_ptr_t)), BLOCK_SIZE*sizeof(key_ptr_t));
    }
}

void load_next(uint32_t in, uint32_t out, uint32_t load, uint32_t count) {
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

int main() {

    uint32_t tasklet_id = me();

    uint32_t size = 524288;
    uint32_t buf_c_custkey = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t buf_o_custkey = (uint32_t) (buf_c_custkey + size*sizeof(key_ptr_t));
    uint32_t buf_c_nationkey = (uint32_t) (buf_o_custkey + size*sizeof(key_ptr_t));
    uint32_t buf_o_orderkey = (uint32_t) (buf_c_nationkey + size*sizeof(key_ptr_t));
    uint32_t buffer_1 = (uint32_t) (buf_o_orderkey + size*sizeof(key_ptr_t));
    uint32_t buffer_2 = (uint32_t) (buffer_1 + size*sizeof(key_ptr_t));
    uint32_t sizes = (uint32_t) (buffer_2 + size*sizeof(key_ptr_t));

    /*
    * JOIN c_custkey = o_custkey
    */
    create_ptr(buf_c_custkey, buf_c_custkey, dpu_args.c_count);

    if (tasklet_id == 0) {
        hash_args.in_ptr = buf_c_custkey;
        hash_args.size = dpu_args.c_count;
        hash_args.shift = 14;
        hash_args.table_ptr = buffer_1;
        hash_args.table_size = 4096;
    }
    barrier_wait(&barrier);

    hash_kernel(&hash_args);
    barrier_wait(&barrier);

    create_ptr(buf_o_custkey, buf_o_custkey, dpu_args.o_count);

    if (tasklet_id == 0) {
        part_args.in_ptr = buf_o_custkey;
        part_args.size = dpu_args.o_count;
        part_args.shift = 14;
        part_args.part_ptr = buffer_2;
        part_args.part_sizes = sizes;
        part_args.part_n = 16;
    }
    barrier_wait(&barrier);

    part_kernel(&part_args);
    barrier_wait(&barrier);

    if (tasklet_id == 0) {
        merge_args.table_ptr = buffer_1;
        merge_args.size = dpu_args.o_count;
        merge_args.in_ptr = buffer_2;
        merge_args.out_ptr = buf_c_custkey;
        merge_args.part_sizes = sizes;
        merge_args.part_n = 16;
    }
    barrier_wait(&barrier);

    merge_kernel(&merge_args, &merge_res);
    barrier_wait(&barrier);

    // Partion o_orderkey for JOIN
    load_next(buf_c_custkey, buffer_1, buf_o_orderkey, merge_res.out_n);

    if (tasklet_id == 0) {
        part_args.in_ptr = buffer_1;
        part_args.size = merge_res.out_n;
        part_args.shift = 27;
        part_args.part_ptr = buf_c_custkey;
        part_args.part_sizes = sizes;
        part_args.part_n = NR_DPU;
    }
    barrier_wait(&barrier);

    part_kernel(&part_args);
    barrier_wait(&barrier);

    // Load c_nationkey for further steps
    load_out(buf_c_custkey, buf_o_custkey, buf_c_nationkey, merge_res.out_n);

    dpu_results.count = merge_res.out_n;

    return 0;
}