#include <defs.h>
#include <barrier.h>
#include <mram.h>
#include <alloc.h>
#include <stdint.h>
#include <stdio.h>
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
part_arguments_t part_args;
match_arguments_t match_args;
merge_arguments_t merge_args;
merge_results_t merge_res;

sel_results_t sel_results;

__mram_noinit_keep uint32_t o_orderkey[524288];
__mram_noinit_keep uint32_t o_orderdate[524288];
__mram_noinit_keep char o_orderpriority[524288][16];

BARRIER_INIT(barrier, NR_TASKLETS);

MUTEX_INIT(mutex);
MUTEX_POOL_INIT(mutexes, 16);

void create_ptr(uint32_t in_key, uint32_t out, uint32_t count) {

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
        mram_read((__mram_ptr void*) (in_key + base*sizeof(uint32_t)), key_cache, BLOCK_SIZE*sizeof(uint32_t));

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

void load_out(uint32_t in, uint32_t out, uint32_t count, uint32_t load) {

    uint32_t tasklet_id = me();

    key_ptr32* ptr_cache = (key_ptr32*) mem_alloc(16*sizeof(key_ptr32));
    char* out_cache = (char*) mem_alloc(16*16);

    uint32_t base_tasklet = tasklet_id*16;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*16) {
        uint32_t size_load = base + 16 > count ? count % 16 : 16;
        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr32)), ptr_cache, size_load*sizeof(key_ptr32));

        for (uint32_t i = 0; i < size_load; i++) {
            mram_read((__mram_ptr void*) (load + ptr_cache[i].ptr*16), &out_cache[i*16], 16);
        }

        mram_write(out_cache, (__mram_ptr void*) (out + base*16), size_load*16);
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

bool pred_o(key_ptr_t element) {
    return element.key >= dpu_args.date_start && element.key < dpu_args.date_end;
}

int main() {

    uint32_t tasklet_id = me();

    uint32_t size = 524288;
    uint32_t buffer_1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t buffer_2 = (uint32_t) (buffer_1 + size*sizeof(key_ptr_t));
    uint32_t sizes = (uint32_t) (buffer_2 + size*sizeof(key_ptr_t));


    create_ptr((uint32_t) o_orderdate, buffer_2, dpu_args.o_count);
    barrier_wait(&barrier);

    sel_arguments_t sel_args = {.in = buffer_2, .out = buffer_1, .pred = &pred_o, .size = dpu_args.o_count};
    sel_kernel(&sel_args, &sel_results);
    barrier_wait(&barrier);

    load_next(buffer_1, buffer_2, (uint32_t) o_orderkey, sel_results.t_count);

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

    load_out(buffer_1, buffer_2, sel_results.t_count, (uint32_t) o_orderpriority);

    dpu_results.count = sel_results.t_count;

    return 0;
}