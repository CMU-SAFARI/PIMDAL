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
#include "hash_join.h"
#include "aggregate.h"

#define BLOCK_SIZE 64
#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif
#ifndef NR_DPU
#define NR_DPU 4
#endif

__host query_args_t dpu_args;
__host query_res_t dpu_results;

aggr_results_t aggr_res;

hash_arguments_t hash_args;
part_arguments_t part_args;
match_arguments_t match_args;
merge_arguments_t merge_args;
merge_results_t merge_res;

BARRIER_INIT(barrier, NR_TASKLETS);

MUTEX_INIT(mutex);
MUTEX_POOL_INIT(mutexes, 16);

void create_ptr(uint32_t in, uint32_t in_nationkey, uint32_t out, uint32_t count) {

    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptr32* ptr_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));
    uint32_t* key_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr32)), ptr_cache, BLOCK_SIZE*sizeof(key_ptr32));
        mram_read((__mram_ptr void*) (in_nationkey + base*sizeof(uint32_t)), key_cache, BLOCK_SIZE*sizeof(uint32_t));

        //printf("%u\n", key_cache[0]);
        for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
            ptr_cache[i].key = (ptr_cache[i].key << 8) | key_cache[i];
            ptr_cache[i].ptr = base+i;
        }
        //printf("%u\n", ptr_cache[0].key);

        mram_write(ptr_cache, (__mram_ptr void*) (out + base*sizeof(key_ptr32)), BLOCK_SIZE*sizeof(key_ptr32));
    }
}

void load_next(uint32_t in, uint32_t load_nationkey, uint32_t load_price,
               uint32_t load_discount, uint32_t out, uint32_t count) {

    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptr32* ptr_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));
    key_ptr_t* out_cache = (key_ptr_t*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        uint32_t size_load = base + BLOCK_SIZE > count ? count % BLOCK_SIZE : BLOCK_SIZE;
        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr32)), ptr_cache, BLOCK_SIZE*sizeof(key_ptr32));

        //printf("%u\n", key_cache[0]);
        for (uint32_t i = 0; i < size_load; i++) {
            __dma_aligned uint32_t read[2];
            uint32_t offset = ptr_cache[i].key & 1;
            uint32_t addr = ptr_cache[i].key - offset;
            mram_read((__mram_ptr void*) (load_nationkey + addr*sizeof(uint32_t)), read, 2*sizeof(uint32_t));
            out_cache[i].key = read[offset];

            __dma_aligned int64_t l_extendedprice;
            mram_read((__mram_ptr void*) (load_price + ptr_cache[i].key*sizeof(int64_t)), &l_extendedprice, sizeof(int64_t));

            __dma_aligned int64_t l_discount;
            mram_read((__mram_ptr void*) (load_discount + ptr_cache[i].key*sizeof(int64_t)), &l_discount, sizeof(int64_t));

            out_cache[i].revenue = l_extendedprice * (100 - l_discount);

        }

        mram_write(out_cache, (__mram_ptr void*) (out + base*sizeof(key_ptr_t)), BLOCK_SIZE*sizeof(key_ptr_t));
    }
}

void load_out(uint32_t in, uint32_t key_out, uint32_t rev_out, uint32_t count) {

    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptr_t* ptr_cache = (key_ptr_t*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));
    uint32_t* key_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));
    int64_t* rev_cache = (int64_t*) mem_alloc(BLOCK_SIZE*sizeof(int64_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        uint32_t size_load = base + BLOCK_SIZE > count ? count % BLOCK_SIZE : BLOCK_SIZE;
        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr_t)), ptr_cache, BLOCK_SIZE*sizeof(key_ptr_t));

        //printf("%u\n", key_cache[0]);
        for (uint32_t i = 0; i < size_load; i++) {
            key_cache[i] = ptr_cache[i].key;
            rev_cache[i] = ptr_cache[i].revenue;
        }

        mram_write(key_cache, (__mram_ptr void*) (key_out + base*sizeof(uint32_t)), BLOCK_SIZE*sizeof(uint32_t));
        mram_write(rev_cache, (__mram_ptr void*) (rev_out + base*sizeof(int64_t)), BLOCK_SIZE*sizeof(int64_t));
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

key_ptr_t sum(key_ptr_t curr_val, key_ptr_t element) {
    curr_val.revenue += element.revenue;

    return curr_val;
}

int main() {

    uint32_t tasklet_id = me();

    uint32_t size = 524288;
    uint32_t buf_s_suppkey = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t buf_l_suppkey = (uint32_t) (buf_s_suppkey + size*sizeof(key_ptr32));
    uint32_t buf_s_nationkey = (uint32_t) (buf_l_suppkey + size*sizeof(key_ptr32));
    uint32_t buf_l_nationkey = (uint32_t) (buf_s_nationkey + size*sizeof(key_ptr32));
    uint32_t buf_l_extendedprice = (uint32_t) (buf_l_nationkey + size*sizeof(key_ptr32));
    uint32_t buf_l_discount = (uint32_t) (buf_l_extendedprice + size*sizeof(key_ptr32));
    uint32_t buffer_1 = (uint32_t) (buf_l_discount + size*sizeof(key_ptr32));
    uint32_t buffer_2 = (uint32_t) (buffer_1 + size*sizeof(key_ptr32));
    uint32_t sizes = (uint32_t) (buffer_2 + size*sizeof(key_ptr32));

    /*
    * JOIN s_suppkey, s_nationkey = l_supkkey, n_nationkey
    */
    create_ptr(buf_s_suppkey, buf_s_nationkey, buf_s_suppkey, dpu_args.s_count);

    if (tasklet_id == 0) {
        hash_args.in_ptr = buf_s_suppkey;
        hash_args.size = dpu_args.s_count;
        hash_args.shift = 14;
        hash_args.table_ptr = buffer_1;
        hash_args.table_size = 1024;
    }
    barrier_wait(&barrier);

    hash_kernel(&hash_args);
    barrier_wait(&barrier);

    create_ptr(buf_l_suppkey, buf_l_nationkey, buf_l_suppkey, dpu_args.l_count);

    if (tasklet_id == 0) {
        part_args.in_ptr = buf_l_suppkey;
        part_args.size = dpu_args.l_count;
        part_args.shift = 14;
        part_args.part_ptr = buffer_2;
        part_args.part_sizes = sizes;
        part_args.part_n = 4;
    }
    barrier_wait(&barrier);

    part_kernel(&part_args);
    barrier_wait(&barrier);

    if (tasklet_id == 0) {
        merge_args.table_ptr = buffer_1;
        merge_args.size = dpu_args.l_count;
        merge_args.in_ptr = buffer_2;
        merge_args.out_ptr = buf_s_suppkey;
        merge_args.part_sizes = sizes;
        merge_args.part_n = 4;
    }
    barrier_wait(&barrier);

    merge_kernel(&merge_args, &merge_res);
    barrier_wait(&barrier);

    /*
    * AGGREGATE BY n_name, revenue: sum
    */
    load_next(buf_s_suppkey, buf_l_nationkey, buf_l_extendedprice,
              buf_l_discount, buffer_2, merge_res.out_n);
    barrier_wait(&barrier);

    aggr_arguments_t aggr_args = {.in = buffer_2, .out = buffer_1, .size = merge_res.out_n, .aggr = sum};
    group_kernel(&aggr_args, &aggr_res);
    barrier_wait(&barrier);

    load_out(buffer_1, buf_s_suppkey, buf_l_suppkey, aggr_res.t_count);

    dpu_results.count = aggr_res.t_count;

    return 0;
}