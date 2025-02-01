/*
* Select with multiple tasklets
*
*/
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>
#include <mutex.h>

#include "reduce.h"

#define BLOCK_SIZE 64
#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

int64_t sum;

extern barrier_t barrier;
extern mutex_id_t mutex;

/*
    @param cache_A input WRAM cache
    @param size number of elements

    Reduces the elements in the cache to a sum
*/
int64_t reduce_add(key_ptr_t *cache_A, uint32_t size) {
    int64_t sum_local = 0;
    for (uint32_t i = 0; i < size; i++) {
        sum_local += cache_A[i].key;
    }

    return sum_local;
}

// The reduce kernel
int reduce_kernel(red_arguments_t *input_args, red_results_t *result) {
    unsigned int tasklet_id = me();

    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
        sum = 0;
    }
    // Barrier
    barrier_wait(&barrier);

    uint32_t input_size_dpu = input_args->size;
    int64_t sum_local = 0;

    // Address of the current processing block in MRAM
    uint32_t base_tasklet = tasklet_id * BLOCK_SIZE;
    key_ptr_t* mram_base_addr_A = (key_ptr_t*) input_args->in;

    // Initialize a local cache to store the MRAM block
    key_ptr_t *cache_A = (key_ptr_t *) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    for (unsigned int base = base_tasklet; base < input_size_dpu; base += BLOCK_SIZE * NR_TASKLETS) {
        uint32_t size = base + BLOCK_SIZE > input_size_dpu ? input_size_dpu % BLOCK_SIZE : BLOCK_SIZE;
        mram_read((__mram_ptr void*) (mram_base_addr_A + base), cache_A, size*sizeof(key_ptr_t));

        sum_local += reduce_add(cache_A, size);
    }

    mutex_lock(mutex);
    sum += sum_local;
    mutex_unlock(mutex);

    barrier_wait(&barrier);
    result->sum = sum;

    return 0;
}