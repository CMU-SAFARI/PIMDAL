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

    key_ptr32* ptr_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr32)), ptr_cache, BLOCK_SIZE*sizeof(key_ptr32));

        for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
            //printf("%u - %u\n", ptr_cache[i].key, ptr_cache[i].ptr);
            ptr_cache[i].ptr = base+i;
        }
        //printf("%u\n", ptr_cache[0].key);

        mram_write(ptr_cache, (__mram_ptr void*) (out + base*sizeof(key_ptr32)), BLOCK_SIZE*sizeof(key_ptr32));
    }
}

/*
    Load a key from the inner relation and create a new ptr for it.
*/
void load_outer_key(uint32_t in, uint32_t out, uint32_t load, uint32_t count) {
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
            sel_cache[i].ptr = base + i;
        }

        mram_write(sel_cache, (__mram_ptr void*) (out + base*sizeof(key_ptr32)), size_load*sizeof(key_ptr32));
    }
}

/*
    Load a 32 bit datatype from the inner relation.
*/
void load_inner_32(uint32_t in, uint32_t out, uint32_t load, uint32_t count) {
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
        //printf("%u\n", out_cache[0]);

        mram_write(out_cache, (__mram_ptr void*) (out + base*sizeof(uint32_t)), BLOCK_SIZE*sizeof(uint32_t));
    }
}

/*
    Load a 64 bit datatype from the outer relation.
*/
void load_outer_64(uint32_t in, uint32_t out, uint32_t load, uint32_t count) {
    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptr32* sel_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));
    int64_t* out_cache = (int64_t*) mem_alloc(BLOCK_SIZE*sizeof(int64_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        uint32_t size_load = base + BLOCK_SIZE > count ? count % BLOCK_SIZE : BLOCK_SIZE;

        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr32)), sel_cache, size_load*sizeof(key_ptr32));

        for (uint32_t i = 0; i < size_load; i++) {

            mram_read((__mram_ptr void*) (load + sel_cache[i].key*sizeof(int64_t)), &out_cache[i], sizeof(int64_t));
        }

        mram_write(out_cache, (__mram_ptr void*) (out + base*sizeof(int64_t)), size_load*sizeof(int64_t));
    }
}

void load_out_64(uint32_t in, uint32_t out, uint32_t load, uint32_t count) {
    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptr32* sel_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));
    int64_t* out_cache = (int64_t*) mem_alloc(BLOCK_SIZE*sizeof(int64_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        uint32_t size_load = base + BLOCK_SIZE > count ? count % BLOCK_SIZE : BLOCK_SIZE;

        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr32)), sel_cache, size_load*sizeof(key_ptr32));

        for (uint32_t i = 0; i < size_load; i++) {

            mram_read((__mram_ptr void*) (load + sel_cache[i].ptr*sizeof(int64_t)), &out_cache[i], sizeof(int64_t));
        }

        mram_write(out_cache, (__mram_ptr void*) (out + base*sizeof(int64_t)), size_load*sizeof(int64_t));
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

void out_val64(uint32_t in, uint32_t count) {
    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    int64_t* out_cache = (int64_t*) mem_alloc(BLOCK_SIZE*sizeof(int64_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        uint32_t size_load = base + BLOCK_SIZE > count ? count % BLOCK_SIZE : BLOCK_SIZE;

        mram_read((__mram_ptr void*) (in + base*sizeof(int64_t)), out_cache, BLOCK_SIZE*sizeof(int64_t));

        for (uint32_t i = 0; i < size_load; i++) {
            printf("%lld\n", out_cache[i]);
        }
    }
}

int main() {

    uint32_t tasklet_id = me();

    uint32_t size = 524288;
    uint32_t buf_o_orderkey = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t buf_l_orderkey = (uint32_t) (buf_o_orderkey + size*sizeof(key_ptr_t));
    uint32_t buf_o_nationkey = (uint32_t) (buf_l_orderkey + size*sizeof(key_ptr_t));
    uint32_t buf_l_suppkey = (uint32_t) (buf_o_nationkey + size*sizeof(key_ptr_t));
    uint32_t buf_l_extendedprice = (uint32_t) (buf_l_suppkey + size*sizeof(key_ptr_t));
    uint32_t buf_l_discount = (uint32_t) (buf_l_extendedprice + size*sizeof(key_ptr_t));
    uint32_t buffer_1 = (uint32_t) (buf_l_discount + size*sizeof(key_ptr_t));
    uint32_t buffer_2 = (uint32_t) (buffer_1 + size*sizeof(key_ptr_t));
    uint32_t sizes = (uint32_t) (buffer_2 + size*sizeof(key_ptr_t));

    /*
    * JOIN o_orderkey = l_orderkey
    */
    create_ptr(buf_o_orderkey, buf_o_orderkey, dpu_args.o_count);

    if (tasklet_id == 0) {
        hash_args.in_ptr = buf_o_orderkey;
        hash_args.size = dpu_args.o_count;
        hash_args.shift = 14;
        hash_args.table_ptr = buffer_1;
        hash_args.table_size = 8192;
    }
    barrier_wait(&barrier);

    hash_kernel(&hash_args);
    barrier_wait(&barrier);

    create_ptr(buf_l_orderkey, buf_l_orderkey, dpu_args.l_count);

    if (tasklet_id == 0) {
        part_args.in_ptr = buf_l_orderkey;
        part_args.size = dpu_args.l_count;
        part_args.shift = 14;
        part_args.part_ptr = buffer_2;
        part_args.part_sizes = sizes;
        part_args.part_n = 32;
    }
    barrier_wait(&barrier);

    part_kernel(&part_args);
    barrier_wait(&barrier);

    if (tasklet_id == 0) {
        merge_args.table_ptr = buffer_1;
        merge_args.size = dpu_args.l_count;
        merge_args.in_ptr = buffer_2;
        merge_args.out_ptr = buf_o_orderkey;
        merge_args.part_sizes = sizes;
        merge_args.part_n = 32;
    }
    barrier_wait(&barrier);

    merge_kernel(&merge_args, &merge_res);
    barrier_wait(&barrier);

    // Load joined data for further use
    load_outer_64(buf_o_orderkey, buffer_1, buf_l_discount, merge_res.out_n);
    barrier_wait(&barrier);

    load_outer_64(buf_o_orderkey, buffer_2, buf_l_extendedprice, merge_res.out_n);
    barrier_wait(&barrier);

    load_inner_32(buf_o_orderkey, buf_l_discount, buf_o_nationkey, merge_res.out_n);
    barrier_wait(&barrier);

    // Partition l_suppkey for JOIN
    load_outer_key(buf_o_orderkey, buf_l_extendedprice, buf_l_suppkey, merge_res.out_n);
    barrier_wait(&barrier);
    
    if (tasklet_id == 0) {
        part_args.in_ptr = buf_l_extendedprice;
        part_args.size = merge_res.out_n;
        part_args.shift = 27;
        part_args.part_ptr = buf_o_orderkey;
        part_args.part_sizes = sizes;
        part_args.part_n = NR_DPU;
    }
    barrier_wait(&barrier);

    part_kernel(&part_args);
    barrier_wait(&barrier);

    // Load o_nationkey based using partitioning
    load_inner_32(buf_o_orderkey, buf_l_orderkey, buf_l_discount, merge_res.out_n);
    barrier_wait(&barrier);

    // Load l_discount based using partitioning
    load_out_64(buf_o_orderkey, buf_o_nationkey, buffer_1, merge_res.out_n);
    barrier_wait(&barrier);

    // Load l_extendedprice using partitioning
    load_out_64(buf_o_orderkey, buf_l_suppkey, buffer_2, merge_res.out_n);
    barrier_wait(&barrier);

    dpu_results.count = merge_res.out_n;

    return 0;
}