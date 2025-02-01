/*
* Unique with multiple tasklets
*
*/
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <handshake.h>
#include <barrier.h>

#include "aggregate.h"

#define BLOCK_SIZE 128
#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

extern barrier_t barrier;

// Array for communication between adjacent tasklets
uint32_t message[NR_TASKLETS];
key_ptr_t  message_value[NR_TASKLETS];
uint32_t message_offset[NR_TASKLETS];
uint32_t message_partial_count;
key_ptr_t  message_last_from_last;

// AGGREGATE in each tasklet
static unsigned int aggregate(key_ptr_t *data, uint32_t n,
                              key_ptr_t (*aggr)(key_ptr_t curr_val, key_ptr_t element)){
    unsigned int pos = 0;
    for(unsigned int j = 1; j < n; j++) {
        if(data[j].key != data[j - 1].key) {
            pos++;
            data[pos] = data[j];
        }
        else {
            data[pos] = aggr(data[pos], data[j]);
        }
    }
    return pos + 1;
}

// Handshake with adjacent tasklets
static uint3 handshake_sync(key_ptr_t *output, unsigned int l_count, unsigned int tasklet_id,
                            key_ptr_t (*aggr)(key_ptr_t curr_val, key_ptr_t element)){
    unsigned int p_count, o_count, offset;
    // Wait and read message
    if(tasklet_id != 0){
        handshake_wait_for(tasklet_id - 1);
        p_count = message[tasklet_id];
        offset = (message_value[tasklet_id].key == output[0].key)?1:0;
        if (offset) {
            output[0] = aggr(output[0], message_value[tasklet_id]);
        }
        o_count = message_offset[tasklet_id];
    }
    else{
        p_count = 0;
        offset = (message_last_from_last.key == output[0].key)?1:0;
        if (offset) {
            output[0] = aggr(output[0], message_last_from_last);
        }
        o_count = 0;
    }
    // Write message and notify
    if(tasklet_id < NR_TASKLETS - 1){
        message[tasklet_id + 1] = p_count + l_count;
        message_value[tasklet_id + 1] = output[l_count - 1];
        message_offset[tasklet_id + 1] = o_count + offset;
        handshake_notify();
    }
    uint3 result = {p_count, o_count, offset}; 
    return result;
}

/*
    Main kernel for performing aggregation on sorted data
*/
int group_kernel(aggr_arguments_t *input_args, aggr_results_t *result) {
    unsigned int tasklet_id = me();

    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    uint32_t input_size_dpu = input_args->size; // Input size per DPU in bytes

    // Address of the current processing block in MRAM
    uint32_t base_tasklet = tasklet_id * BLOCK_SIZE;
    uint32_t mram_base_addr_A = (uint32_t) input_args->in;
    uint32_t mram_base_addr_B = (uint32_t) input_args->out;

    // Initialize a local cache to store the MRAM block
    key_ptr_t *cache_data = (key_ptr_t *) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    // Initialize shared variable
    if(tasklet_id == NR_TASKLETS - 1){
        message_partial_count = 0;
        message_last_from_last.key = 0xFFFFFFFF; // A value that is not in the input array
    }
    // Barrier
    barrier_wait(&barrier);

    uint32_t i = 0; // Iteration count
    uint32_t base = base_tasklet;
    for(; base < input_size_dpu; base += BLOCK_SIZE * NR_TASKLETS){

        uint32_t size = base + BLOCK_SIZE > input_size_dpu ?
                        input_size_dpu % BLOCK_SIZE : BLOCK_SIZE;

        // Load cache with current MRAM block
        mram_read((__mram_ptr void const*)(mram_base_addr_A + base*sizeof(key_ptr_t)), cache_data, size*sizeof(key_ptr_t));

        // AGGREGATE in each tasklet
        uint32_t l_count = aggregate(cache_data, size, input_args->aggr);

        // Sync with adjacent tasklets
        uint3 po_count = handshake_sync(cache_data, l_count, tasklet_id, input_args->aggr);

        // Write cache to current MRAM block
        uint32_t out_off = (message_partial_count + po_count.x - po_count.y - po_count.z) * sizeof(key_ptr_t);
        mram_write(cache_data, (__mram_ptr void*)(mram_base_addr_B + out_off), l_count * sizeof(key_ptr_t));

        // First
        if(tasklet_id == 0 && i == 0) {
            result->first = cache_data[0];
        }

        // Barrier
        barrier_wait(&barrier);
        
        // Total count in this DPU
        if(tasklet_id == NR_TASKLETS - 1 ||
           base + BLOCK_SIZE >= input_size_dpu) {
            message_last_from_last = cache_data[l_count - 1];
            result->last = cache_data[l_count - 1];
            result->t_count = message_partial_count + po_count.x + l_count - po_count.y - po_count.z;
            message_partial_count = result->t_count;
        }

        i++;
    }

    // Sync the idle tasklets
    uint32_t size = (input_size_dpu + NR_TASKLETS*BLOCK_SIZE - 1) / (NR_TASKLETS*BLOCK_SIZE);
    size *= NR_TASKLETS*BLOCK_SIZE;
    if (base < size) {
        handshake_sync(cache_data, 0, tasklet_id, input_args->aggr);

        barrier_wait(&barrier);
    }

    return 0;
}