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
#include <mutex.h>
#include <handshake.h>

#include "datatype.h"
#include "sort_merge.h"

#define BLOCK_SIZE 64
#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

// Array for communication between adjacent tasklets
uint32_t message[NR_TASKLETS];
uint32_t message_partial_count;

// Barrier
extern barrier_t barrier;

// Handshake with adjacent tasklets
static uint32_t handshake_sync(uint32_t l_count, uint32_t tasklet_id, uint32_t *message){
    uint32_t p_count;
    // Wait and read message
    if(tasklet_id != 0){
        handshake_wait_for(tasklet_id - 1);
        p_count = message[tasklet_id];
    }
    else
        p_count = 0;
    // Write message and notify
    if(tasklet_id < NR_TASKLETS - 1){
        message[tasklet_id + 1] = p_count + l_count;
        handshake_notify();
    }
    return p_count;
}

/*
    @param inner_buf WRAM cache for the inner relation
    @param outer_buf WRAM cache with the elements from outer relation to join
    @param inner pointer to a pointer keeping track of position in outer relation
    @param inner_end end point of inner relation
    @param n number of input elements
*/
uint32_t merge(key_ptr32 *inner_buf, key_ptr32* outer_buf, key_ptr32 **inner,
               key_ptr32 *inner_end, uint32_t n) {
    uint32_t out_i = 0;
    uint32_t inner_i = 0;

    for (uint32_t outer_i = 0; outer_i < n; outer_i++) {
        // Iterate through inner buffer
        while (outer_buf[outer_i].key > inner_buf[inner_i].key) {
            inner_i++;
            if (inner_i == BLOCK_SIZE) {
                // Load new elements into inner buffer
                *inner += BLOCK_SIZE;
                if (*inner >= inner_end) {
                    return out_i;
                }
                mram_read((__mram_ptr void*) *inner, inner_buf, BLOCK_SIZE*sizeof(key_ptr32));
                inner_i = 0;
            }
        }

        if (outer_buf[outer_i].key == inner_buf[inner_i].key) {
            // Match in outer and inner buffer
            outer_buf[out_i].key = outer_buf[outer_i].ptr;
            outer_buf[out_i].ptr = inner_buf[inner_i].ptr;
            out_i++;
        }
    }

    return out_i;
}

// The merge kernel
int merge_kernel(merge_arguments_t *input_args, merge_results_t *result) {
    unsigned int tasklet_id = me();

    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    // Address of the current processing block in MRAM
    key_ptr32* inner_addr = (key_ptr32*) input_args->inner;
    key_ptr32* outer_addr = (key_ptr32*) input_args->outer;
    key_ptr32* out_addr = (key_ptr32*) input_args->out;
    uint32_t n_inner = input_args->size_inner;

    key_ptr32* cache_inner = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));
    key_ptr32* cache_outer = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));

    uint32_t base = tasklet_id * BLOCK_SIZE;

    key_ptr32 *inner = inner_addr;
    mram_read((__mram_ptr void*) inner, cache_inner, BLOCK_SIZE*sizeof(key_ptr32));

    for(; base < input_args->size_outer; base += BLOCK_SIZE * NR_TASKLETS){

        uint32_t size = base + BLOCK_SIZE > input_args->size_outer ?
                        input_args->size_outer % BLOCK_SIZE : BLOCK_SIZE;

        // Load cache with current MRAM block
        mram_read((__mram_ptr void const*)(outer_addr + base), cache_outer, size*sizeof(key_ptr32));

        uint32_t l_count = merge(cache_inner, cache_outer, &inner, inner_addr+n_inner, size);
        // Sync with adjacent tasklets
        uint32_t p_count = handshake_sync(l_count, tasklet_id, message);

        barrier_wait(&barrier);

        // Write cache to current MRAM block
        uint32_t out_off = message_partial_count + p_count;
        if (l_count > 0) {
            mram_write(cache_outer, (__mram_ptr void*)(out_addr + out_off), l_count * sizeof(key_ptr32));
        }

        // Total count in this DPU
        if(tasklet_id == NR_TASKLETS - 1){
            result->t_count = message_partial_count + p_count + l_count;
            message_partial_count = result->t_count;
        }

    }

    // Sync the not working tasklets
    uint32_t size = (input_args->size_outer + NR_TASKLETS*BLOCK_SIZE - 1) / (NR_TASKLETS*BLOCK_SIZE);
    size *= NR_TASKLETS*BLOCK_SIZE;
    if (base < size) {
        uint32_t p_count = handshake_sync(0, tasklet_id, message);

        barrier_wait(&barrier);

        // Total count in this DPU
        if(tasklet_id == NR_TASKLETS - 1){
            result->t_count = message_partial_count + p_count;
        }
    }
 
    return 0;
}
