/*
* Select with multiple tasklets
*
*/
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>

#include "arithmetic.h"

#define BLOCK_SIZE 64
#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

extern barrier_t barrier;

/*
    @param cache_A first input and output
    @param cache_B second input
    @param size number of elements

    Multiply the elements in cache_A by the elements in cache_B
*/
void mul(key_ptr_t *cache_A, key_ptr_t *cache_B, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        cache_A[i].key *= cache_B[i].key;
    }
}

// The mul kernel
int arithmetic_kernel(ar_arguments_t *input_args) {
    unsigned int tasklet_id = me();

    if (tasklet_id == 0){ // Initialize once the cycle counter
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    uint32_t input_size_dpu = input_args->size;

    // Address of the current processing block in MRAM
    uint32_t base_tasklet = tasklet_id * BLOCK_SIZE;
    key_ptr_t* mram_base_addr_A = (key_ptr_t*) input_args->buffer_A;
    key_ptr_t* mram_base_addr_B = (key_ptr_t*) input_args->buffer_B;
    key_ptr_t* mram_base_addr_out = (key_ptr_t*) input_args->buffer_out;

    // Initialize a local cache to store the MRAM block
    key_ptr_t *cache_A = (key_ptr_t *) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));
    key_ptr_t *cache_B = (key_ptr_t *) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    for (unsigned int base = base_tasklet; base < input_size_dpu; base += BLOCK_SIZE * NR_TASKLETS) {
        uint32_t size = base + BLOCK_SIZE <= input_size_dpu ? BLOCK_SIZE : input_size_dpu % BLOCK_SIZE;
        mram_read((__mram_ptr void*) (mram_base_addr_A + base), cache_A, size*sizeof(key_ptr_t));
        mram_read((__mram_ptr void*) (mram_base_addr_B + base), cache_B, size*sizeof(key_ptr_t));

        mul(cache_A, cache_B, size);

        mram_write(cache_A, (__mram_ptr void*) (mram_base_addr_out + base), size*sizeof(key_ptr_t));
    }

    return 0;
}
