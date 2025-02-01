#include <defs.h>
#include <barrier.h>
#include <mram.h>
#include <alloc.h>
#include <stdint.h>
#include <stdio.h>
#include <mutex.h>
#include <mutex_pool.h>
#include <perfcounter.h>

#include "groupby.h"
#include "datatype.h"
#include "aggregate.h"

#define BLOCK_SIZE 64
#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif

__host groupby_arguments_t dpu_args;
__host groupby_results_t dpu_results;
__host uint64_t cycles;

aggr_results_t proj_res;

BARRIER_INIT(barrier, NR_TASKLETS);

MUTEX_INIT(mutex);
MUTEX_POOL_INIT(mutexes, 16);

/*
    Create a key value type for sum aggregation
*/
void create_sumptr(uint32_t *in_key, uint32_t *in_val, key_ptr_t *out) {
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
        key_ptr_t new_ptr = {.key = in_key[i], .ptr = in_val[i]};
        out[i] = new_ptr;
    }
}

/*
    Create a key value type for unique or count aggregation
*/
void create_ptr(uint32_t *in_key, key_ptr_t *out) {
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
        key_ptr_t new_ptr = {.key = in_key[i], .ptr = 1};
        out[i] = new_ptr;
    }
}

/*
    Output the key of the unique aggregation
*/
void out_key(key_ptr_t *out, uint32_t *key) {
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
        key[i] = out[i].key;
    }
}

/*
    Output the key and value of the aggregation
*/
void out_keyval(key_ptr_t *out, uint32_t *key, uint32_t *val) {
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
        key[i] = out[i].key;
        val[i] = out[i].ptr;
    }
}

/*
    Aggregation function unique
*/
key_ptr_t unique(key_ptr_t curr_val, __attribute__((unused)) key_ptr_t element) {

    return curr_val;
}

/*
    Aggregation function sum
*/
key_ptr_t sum(key_ptr_t curr_val, key_ptr_t element) {
    curr_val.ptr += element.ptr;

    return curr_val;
}

int main() {

    uint32_t tasklet_id = me();

    uint32_t offset = dpu_args.size + (dpu_args.size & 1);
    uint32_t keys = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t vals = keys + offset*sizeof(uint32_t);
    uint32_t keys_out = vals + offset*sizeof(uint32_t);
    uint32_t vals_out = keys_out + offset*sizeof(uint32_t);

    uint32_t* key_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));
    uint32_t* val_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));
    key_ptr_t* out_cache = (key_ptr_t*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < dpu_args.size; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (keys + base*sizeof(uint32_t)), key_cache, BLOCK_SIZE*sizeof(uint32_t));

        if (dpu_args.kernel_sel == 2) {
            mram_read((__mram_ptr void*) (vals + base*sizeof(uint32_t)), val_cache, BLOCK_SIZE*sizeof(uint32_t));
            create_sumptr(key_cache, val_cache, out_cache);
        }
        else {
            create_ptr(key_cache, out_cache);
        }

        mram_write(out_cache, (__mram_ptr void*) (keys_out + base*sizeof(key_ptr_t)), BLOCK_SIZE*sizeof(key_ptr_t));
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

    aggr_arguments_t proj_args = {.in = keys_out, .out = keys, .size = dpu_args.size, .aggr = unique};
    if (dpu_args.kernel_sel > 0) {
        proj_args.aggr = sum;
    }
    group_kernel(&proj_args, &proj_res);

#if PERF > 0
    if (tasklet_id == 0) {
        cycles = perfcounter_get();
    }
    barrier_wait(&barrier);
#endif

    if (tasklet_id == 0){
        dpu_results.count = proj_res.t_count;
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    out_cache = (key_ptr_t*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));
    val_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));
    key_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));

    base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < proj_res.t_count; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (keys + base*sizeof(key_ptr_t)), out_cache, BLOCK_SIZE*sizeof(key_ptr_t));

        if (dpu_args.kernel_sel > 0) {
            out_keyval(out_cache, key_cache, val_cache);
            mram_write(val_cache, (__mram_ptr void*) (vals_out + base*sizeof(uint32_t)), BLOCK_SIZE*sizeof(uint32_t));
        }
        else {
            out_key(out_cache, key_cache);
        }

        mram_write(key_cache, (__mram_ptr void*) (keys_out + base*sizeof(uint32_t)), BLOCK_SIZE*sizeof(uint32_t));
    }

    barrier_wait(&barrier);

    return 0;
}