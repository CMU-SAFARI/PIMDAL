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
#include "sort.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif
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
merge_arguments_t merge_args;
merge_results_t merge_res;

aggr_results_t proj_res;

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
            ptr_cache[i].ptr = base+i;
        }
        //printf("%u\n", ptr_cache[0].key);

        mram_write(ptr_cache, (__mram_ptr void*) (out + base*sizeof(key_ptr32)), BLOCK_SIZE*sizeof(key_ptr32));
    }
}

void create_outptr(uint32_t in, uint32_t out, uint32_t load_key, uint32_t load_date,
                   uint32_t load_prio, uint32_t load_price, uint32_t load_discount,
                   uint32_t count) {

    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptr32* sel_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));
    key_ptr_t* out_cache = (key_ptr_t*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        uint32_t size_load = base + BLOCK_SIZE > count ? count % BLOCK_SIZE : BLOCK_SIZE;

        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr32)), sel_cache, size_load*sizeof(key_ptr32));

        for (uint32_t i = 0; i < size_load; i++) {
            __dma_aligned uint32_t read[2];
            uint32_t offset = sel_cache[i].ptr & 1;
            uint32_t addr = sel_cache[i].ptr - offset;

            mram_read((__mram_ptr void*) (load_key + sel_cache[i].key*sizeof(key_ptr32)), read, 2*sizeof(uint32_t));
            out_cache[i].orderkey = read[0];
            
            mram_read((__mram_ptr void*) (load_date + addr*sizeof(uint32_t)), read, 2*sizeof(uint32_t));
            out_cache[i].orderdate = read[offset];

            mram_read((__mram_ptr void*) (load_prio + addr*sizeof(uint32_t)), read, 2*sizeof(uint32_t));
            out_cache[i].shippriority = read[offset];

            __dma_aligned int64_t extendedprice;
            mram_read((__mram_ptr void*) (load_price + sel_cache[i].key*sizeof(int64_t)), &extendedprice, sizeof(int64_t));

            __dma_aligned int64_t discount;
            mram_read((__mram_ptr void*) (load_discount + sel_cache[i].key*sizeof(int64_t)), &discount, sizeof(int64_t));

            out_cache[i].key = extendedprice * (100 - discount);

            // if (tasklet_id == 0)
            // printf("%u - %lld - %u - %u\n", out_cache[i].key, out_cache[i].revenue,
            //        out_cache[i].orderdate, out_cache[i].shippriority);
        }
        //printf("%u - %lld - %u - %u\n", out_cache[0].key, out_cache[0].revenue,
        //           out_cache[0].orderdate, out_cache[0].shippriority);

        mram_write(out_cache, (__mram_ptr void*) (out + base*sizeof(key_ptr_t)), size_load*sizeof(key_ptr_t));
    }
}

void load_out(uint32_t in, uint32_t out, uint32_t count) {
    
    mem_reset(); // Reset the heap

    key_ptr_t* in_cache = (key_ptr_t*) mem_alloc(10*sizeof(key_ptr_t));
    uint32_t* key_cache = (uint32_t*) mem_alloc(10*sizeof(uint32_t));
    int64_t* rev_cache = (int64_t*) mem_alloc(10*sizeof(int64_t));
    uint32_t* date_cache = (uint32_t*) mem_alloc(10*sizeof(uint32_t));
    uint32_t* prio_cache = (uint32_t*) mem_alloc(10*sizeof(uint32_t));
    
    uint32_t end = in + (count - 10) * sizeof(key_ptr_t);
    mram_read((__mram_ptr void*) end, in_cache, 10*sizeof(key_ptr_t));

    for (uint32_t i = 0; i < 10; i++) {
        key_cache[i] = in_cache[9-i].orderkey;
        rev_cache[i] = in_cache[9-i].key;
        date_cache[i] = in_cache[9-i].orderdate;
        prio_cache[i] = in_cache[9-i].shippriority;

        //printf("%u %lld %u %u\n", key_cache[i], rev_cache[i], date_cache[i], prio_cache[i]);
    }

    mram_write(key_cache, (__mram_ptr void*) (out), 10*sizeof(uint32_t));
    mram_write(rev_cache, (__mram_ptr void*) (out + 16*sizeof(key_ptr32)), 10*sizeof(int64_t));
    mram_write(date_cache, (__mram_ptr void*) (out + 32*sizeof(key_ptr32)), 10*sizeof(uint32_t));
    mram_write(prio_cache, (__mram_ptr void*) (out + 48*sizeof(key_ptr32)), 10*sizeof(uint32_t));
}

void out_val(uint32_t in, uint32_t count) {
    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptr_t* out_cache = (key_ptr_t*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        uint32_t size_load = base + BLOCK_SIZE > count ? count % BLOCK_SIZE : BLOCK_SIZE;

        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr_t)), out_cache, size_load*sizeof(key_ptr_t));

        for (uint32_t i = 0; i < size_load; i++) {
            printf("%u - %lld - %u - %u\n", out_cache[i].orderkey, out_cache[i].key,
                    out_cache[i].orderdate, out_cache[i].shippriority);
        }
    }
}

/*
    Aggregation function sum
*/
key_ptr_t sum(key_ptr_t curr_val, key_ptr_t element) {
    curr_val.key += element.key;

    return curr_val;
}

int main() {

    uint32_t tasklet_id = me();

    uint32_t size = 524288;
    uint32_t buf_o_orderkey = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t buf_l_orderkey = (uint32_t) (buf_o_orderkey + size*sizeof(key_ptr32));
    uint32_t buf_o_orderdate = (uint32_t) (buf_l_orderkey + size*sizeof(key_ptr32));
    uint32_t buf_o_shipprio = (uint32_t) (buf_o_orderdate + size*sizeof(key_ptr32));
    uint32_t buf_l_extendedprice = (uint32_t) (buf_o_shipprio + size*sizeof(key_ptr32));
    uint32_t buf_l_discount = (uint32_t) (buf_l_extendedprice + size*sizeof(key_ptr32));
    uint32_t buffer_1 = (uint32_t) (buf_l_discount + size*sizeof(key_ptr32));
    uint32_t buffer_2 = (uint32_t) (buffer_1 + size*sizeof(key_ptr32));
    uint32_t buffer_3 = (uint32_t) (buffer_2 + size*sizeof(key_ptr32));
    uint32_t sizes_1 = (uint32_t) (buffer_3 + size*sizeof(key_ptr32));

    create_ptr(buf_o_orderkey, buf_o_orderkey, dpu_args.o_count);

    if (tasklet_id == 0) {
        hash_args.in_ptr = buf_o_orderkey;
        hash_args.size = dpu_args.o_count;
        hash_args.shift = 13;
        hash_args.table_ptr = buffer_1;
        hash_args.table_size = 16384;
    }
    barrier_wait(&barrier);

    hash_kernel(&hash_args);
    barrier_wait(&barrier);

    create_ptr(buf_l_orderkey, buffer_3, dpu_args.l_count);

    if (tasklet_id == 0) {
        part_args.in_ptr = buffer_3;
        part_args.size = dpu_args.l_count;
        part_args.shift = 13;
        part_args.part_ptr = buffer_2;
        part_args.part_sizes = sizes_1;
        part_args.part_n = 64;
    }
    barrier_wait(&barrier);

    part_kernel(&part_args);
    barrier_wait(&barrier);

    if (tasklet_id == 0) {
        merge_args.table_ptr = buffer_1;
        merge_args.size = dpu_args.l_count;
        merge_args.in_ptr = buffer_2;
        merge_args.out_ptr = buf_o_orderkey;
        merge_args.part_sizes = sizes_1;
        merge_args.part_n = 64;
    }
    barrier_wait(&barrier);

    merge_kernel(&merge_args, &merge_res);
    barrier_wait(&barrier);

    create_outptr(buf_o_orderkey, buffer_2, buf_l_orderkey, buf_o_orderdate, buf_o_shipprio,
                  buf_l_extendedprice, buf_l_discount, merge_res.out_n);

    aggr_arguments_t proj_args = {.in = buffer_2, .out = buffer_1, .size = merge_res.out_n, .aggr = sum};
    group_kernel(&proj_args, &proj_res);

    sort_arguments_t sort_args = {.in = buffer_1, 
                                  .nr_splits = 16,
                                  .nr_elements = proj_res.t_count,
                                  .indices = 0, 
                                  .out = buffer_2,
                                  .pivot = {.key = 4000000000},
                                  .start = {.key = 0}};
    barrier_wait(&barrier);

    sort_kernel(&sort_args);
    barrier_wait(&barrier);

    if(tasklet_id == 0) {
        load_out(buffer_2, buf_o_orderkey, proj_res.t_count);
    }

    dpu_results.count = 10;

    return 0;
}