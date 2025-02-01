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

key_ptr_t sum(key_ptr_t curr_val, key_ptr_t element) {
    curr_val.val += element.val;

    return curr_val;
}

void load_out(uint32_t in, uint32_t prio_out, uint32_t count_out, uint32_t count) {

    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptr_t* ptr_cache = (key_ptr_t*) mem_alloc(16*sizeof(key_ptr_t));
    char* prio_cache = (char*) mem_alloc(16*16);
    uint32_t* count_cache = (uint32_t*) mem_alloc(16*sizeof(uint32_t));

    uint32_t base_tasklet = tasklet_id*16;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*16) {
        uint32_t size_load = base + 16 > count ? count % 16 : 16;
        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr_t)), ptr_cache, size_load*sizeof(key_ptr_t));

        //printf("%u\n", key_cache[0]);
        for (uint32_t i = 0; i < size_load; i++) {
            memcpy(&prio_cache[i*16], &ptr_cache[i].prio, 16);
            count_cache[i] = ptr_cache[i].val;
        }

        mram_write(prio_cache, (__mram_ptr void*) (prio_out + base*16), 16*16);
        mram_write(count_cache, (__mram_ptr void*) (count_out + base*sizeof(uint32_t)), 16*sizeof(uint32_t));
    }
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

    if (tasklet_id == 0) {
        uint64_t count;
        mram_read((__mram_ptr void*) buffer_1, &count, sizeof(uint64_t));
        dpu_results.count = count;
    }
    barrier_wait(&barrier);

    aggr_arguments_t aggr_args = {.in = buffer_4, .out = buffer_3, .size = dpu_results.count, .aggr = sum};
    group_kernel(&aggr_args, &aggr_res);

    load_out(buffer_3, buffer_1, buffer_2, aggr_res.t_count);

    dpu_results.count = aggr_res.t_count;

    return 0;
}