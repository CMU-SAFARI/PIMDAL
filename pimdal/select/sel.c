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
#include <handshake.h>

#include "sel.h"

#define BLOCK_SIZE (SEL_BLOCK_BYTES/sizeof(key_ptr_t))
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

// SEL in each tasklet
static unsigned int select(key_ptr_t *data, bool (*pred)(key_ptr_t), uint32_t size){
    unsigned int pos = 0;
    for(unsigned int j = 0; j < size; j++) {
        if((*pred)(data[j])) {
            data[pos] = data[j];
            pos++;
        }
    }
    return pos;
}

// The sel kernel
int sel_kernel(sel_arguments_t *input_args, sel_results_t *result) {
    unsigned int tasklet_id = me();
#if PRINT
    printf("tasklet_id = %u\n", tasklet_id);
#endif
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
        message_partial_count = 0;
    }
    // Barrier
    barrier_wait(&barrier);

    uint32_t input_size_dpu = input_args->size;

    // Address of the current processing block in MRAM
    uint32_t base_tasklet = tasklet_id * BLOCK_SIZE;
    key_ptr_t* mram_base_addr_A = (key_ptr_t*) input_args->in;
    key_ptr_t* mram_base_addr_B = (key_ptr_t*) input_args->out;

    // Initialize a local cache to store the MRAM block
    key_ptr_t *cache_sel = (key_ptr_t *) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    uint32_t base = base_tasklet;
    for(; base < input_size_dpu; base += BLOCK_SIZE * NR_TASKLETS) {
        
        uint32_t size = base + BLOCK_SIZE > input_size_dpu ?
                        input_size_dpu % BLOCK_SIZE : BLOCK_SIZE;

        // Load cache with current MRAM block
        mram_read((__mram_ptr void const*)(mram_base_addr_A + base), cache_sel, size*sizeof(key_ptr_t));

        // SELECT in each tasklet
        uint32_t l_count = select(cache_sel, input_args->pred, size);

        // Sync with adjacent tasklets
        uint32_t p_count = handshake_sync(l_count, tasklet_id, message);

        // Write cache to current MRAM block
        uint32_t out_off = message_partial_count + p_count;
        if (l_count > 0) {
            mram_write(cache_sel, (__mram_ptr void*)(mram_base_addr_B + out_off), l_count * sizeof(key_ptr_t));
        }

        // Barrier
        barrier_wait(&barrier);

        // Total count in this DPU
        if(tasklet_id == NR_TASKLETS - 1){
            result->t_count = message_partial_count + p_count + l_count;
            message_partial_count = result->t_count;
        }

    }

    // Sync the idle tasklets
    uint32_t size = (input_size_dpu + NR_TASKLETS*BLOCK_SIZE - 1) / (NR_TASKLETS*BLOCK_SIZE);
    size *= NR_TASKLETS*BLOCK_SIZE;
    if (base < size) {
        uint32_t p_count = handshake_sync(0, tasklet_id, message);

        barrier_wait(&barrier);

        if(tasklet_id == NR_TASKLETS - 1){
            result->t_count = message_partial_count + p_count;
        }
    }


    return 0;
}
