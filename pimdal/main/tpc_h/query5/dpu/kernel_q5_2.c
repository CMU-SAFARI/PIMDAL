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

__mram_noinit_keep uint32_t o_orderkey[131072];
__mram_noinit_keep uint32_t o_custkey[131072];
__mram_noinit_keep uint32_t o_orderdate[131072];

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

    uint32_t* key_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));
    key_ptr_t* ptr_cache = (key_ptr_t*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (in + base*sizeof(uint32_t)), key_cache, BLOCK_SIZE*sizeof(uint32_t));

        for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
            key_ptr_t new_ptr = {.key = key_cache[i], .ptr = base+i};
            ptr_cache[i] = new_ptr;
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

    key_ptr_t* sel_cache = (key_ptr_t*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        uint32_t size_load = base + BLOCK_SIZE > count ? count % BLOCK_SIZE : BLOCK_SIZE;

        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr_t)), sel_cache, size_load*sizeof(key_ptr_t));

        for (uint32_t i = 0; i < size_load; i++) {
            __dma_aligned uint32_t read[2];
            uint32_t offset = sel_cache[i].ptr & 1;
            uint32_t addr = sel_cache[i].ptr - offset;
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

bool pred_o(key_ptr_t element) {
    bool res = element.key >= dpu_args.date_start &&
        element.key < dpu_args.date_end;
    return res;
}

int main() {

    uint32_t tasklet_id = me();

    uint32_t size = 524288;
    uint32_t buffer_1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t buffer_2 = (uint32_t) (buffer_1 + size*sizeof(key_ptr_t));
    uint32_t buffer_3 = (uint32_t) (buffer_2 + size*sizeof(key_ptr_t));
    uint32_t buffer_4 = (uint32_t) (buffer_3 + size*sizeof(key_ptr_t));
    uint32_t sizes = (uint32_t) (buffer_4 + size*sizeof(key_ptr_t));

    /*
    * o_orderdate < DATE and o_orderadate < DATE + 1 year
    */
    create_ptr((uint32_t) o_orderdate, buffer_2, dpu_args.o_count);
    barrier_wait(&barrier);

    sel_arguments_t sel_args = {.in = buffer_2, .out = buffer_1, .pred = &pred_o, .size = dpu_args.o_count};
    sel_kernel(&sel_args, &sel_results);
    barrier_wait(&barrier);

    // Partition o_custkey for JOIN
    load_next(buffer_1, buffer_2, (uint32_t) o_custkey, sel_results.t_count);

    if (tasklet_id == 0) {
        part_args.in_ptr = buffer_2;
        part_args.size = sel_results.t_count;
        part_args.shift = 27;
        part_args.part_ptr = buffer_1;
        part_args.part_sizes = sizes;
        part_args.part_n = NR_DPU;
    }
    barrier_wait(&barrier);

    part_kernel(&part_args);
    barrier_wait(&barrier);

    // Load o_orderkey using partitioning
    load_out(buffer_1, buffer_2, (uint32_t) o_orderkey, sel_results.t_count);

    dpu_results.count = sel_results.t_count;

    return 0;
}