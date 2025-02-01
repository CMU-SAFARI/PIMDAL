#include <defs.h>
#include <barrier.h>
#include <mram.h>
#include <alloc.h>
#include <stdint.h>
#include <stdio.h>
#include <mutex.h>
#include <string.h>
#include <perfcounter.h>

#include "datatype.h"
#include "sel.h"
#include "select.h"

#ifndef SEL_BLOCK_BYTES
#define SEL_BLOCK_BYTES 1024
#endif
#define BLOCK_SIZE (SEL_BLOCK_BYTES/sizeof(key_ptr_t))
#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif

__host select_arguments_t dpu_args;
__host select_results_t dpu_results;
__host uint64_t cycles;

sel_results_t sel_res;

BARRIER_INIT(barrier, NR_TASKLETS);

void create_ptr(int32_t *in, key_ptr_t *out, uint32_t start) {
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
        key_ptr_t new_ptr = {.key = in[i], .ptr = start+i};
        out[i] = new_ptr;
    }
}

void out_val(key_ptr_t *in, int32_t *out) {
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
        out[i] = in[i].key;
    }
}

bool pred1(key_ptr_t element) {
    return element.key >= 10 && element.key <= 20;
}

int main() {

    uint32_t tasklet_id = me();

    uint32_t buffer_1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t buffer_2 = (uint32_t) (DPU_MRAM_HEAP_POINTER + dpu_args.size*sizeof(key_ptr_t));

    int32_t* val_cache = (int32_t*) mem_alloc(BLOCK_SIZE*sizeof(int32_t));
    key_ptr_t* ptr_cache = (key_ptr_t*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < dpu_args.size; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (buffer_1 + base*sizeof(int32_t)), val_cache, BLOCK_SIZE*sizeof(int32_t));

        create_ptr(val_cache, ptr_cache, base);

        mram_write(ptr_cache, (__mram_ptr void*) (buffer_2 + base*sizeof(key_ptr_t)), BLOCK_SIZE*sizeof(key_ptr_t));
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

    sel_arguments_t sel_args = {.in = buffer_2, .out = buffer_1, .pred = &pred1, .size = dpu_args.size};
    sel_kernel(&sel_args, &sel_res);
    barrier_wait(&barrier);
    //printf("Res: %u\n", sel_res.t_count);

#if PERF > 0
    if (tasklet_id == 0) {
        cycles = perfcounter_get();
    }
    barrier_wait(&barrier);
#endif

    if (tasklet_id == 0){
        dpu_results.count = sel_res.t_count;
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    val_cache = (int32_t*) mem_alloc(BLOCK_SIZE*sizeof(int32_t));
    ptr_cache = (key_ptr_t*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    for (uint32_t base = base_tasklet; base < sel_res.t_count; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (buffer_1 + base*sizeof(key_ptr_t)), ptr_cache, BLOCK_SIZE*sizeof(key_ptr_t));

        out_val(ptr_cache, val_cache);

        mram_write(val_cache, (__mram_ptr void*) (buffer_2 + base*sizeof(int32_t)), BLOCK_SIZE*sizeof(int32_t));
    }

    return 0;
}