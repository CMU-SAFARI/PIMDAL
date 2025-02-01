#include <defs.h>
#include <barrier.h>
#include <mram.h>
#include <alloc.h>
#include <stdint.h>
#include <stdio.h>
#include <mutex_pool.h>
#include <string.h>

#include "aggregate.h"
#include "hash_aggr.h"

#ifndef AGG_BLOCK_BYTES
#define AGG_BLOCK_BYTES 512
#endif
#define BLOCK_SIZE (AGG_BLOCK_BYTES/sizeof(key_ptr_t))
#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif
#ifndef AGG_TABLE_SIZE
#define AGG_TABLE_SIZE 4096
#endif
#define NR_BUCKETS 16
#define BUCKET_SIZE (AGG_TABLE_SIZE/NR_BUCKETS)

key_ptr_t* table;
uint32_t out_pos = 0;
uint32_t wb_pos;

extern struct mutex_pool mutexes;
extern const mutex_id_t mutex;

// Barrier
extern barrier_t barrier;

/*
    @param in WRAM cache of elements to aggregate
    @param n number of elements
    @param aggr aggregation function

    Aggregate the elements into the shared hash table
*/
uint32_t hash_insert(key_ptr_t* in, uint32_t n,
                 key_ptr_t (*aggr)(key_ptr_t curr_val, key_ptr_t element)) {
    
    // Index for writing back not inserted elements
    uint32_t wb_i = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t pos = hash0(in[i]);
        // Bucket to insert
        uint32_t bucket = (pos / BUCKET_SIZE) % NR_BUCKETS;
        // Position inside bucket
        pos = pos % BUCKET_SIZE;

        mutex_pool_lock(&mutexes, bucket);
        bool success = false;
        for (uint32_t k = 0; k < 10; k++) {
            uint32_t index = bucket*BUCKET_SIZE + pos;
            if (empty(table[index])) {
                table[index] = in[i];
                success = true;
                break;
            }
            else if (duplicate(in[i], table[index])) {
                table[index] = aggr(table[index], in[i]);
                success = true;
                break;
            }

            pos = (pos + k) % BUCKET_SIZE;
        }
        mutex_pool_unlock(&mutexes, bucket);

        // Write back element if not inserted
        if (!success) {
            in[wb_i] = in[i];
            wb_i++;
        }
    }

    return wb_i;
}

/*
    @param in_cache WRAM cache to write aggregated elements to output
    @param in_table hash table where elements where aggregated
    @param out the output array in MRAM

    Write the aggreated elements in the shared table to the output
*/
void output(key_ptr_t* in_cache, key_ptr_t* in_table, key_ptr_t* out) {
    uint32_t out_i = 0;
    for (uint32_t i = 0; i < BUCKET_SIZE; i++) {
        if (!empty(in_table[i])) {
            in_cache[out_i] = in_table[i];
            out_i++;

            if (out_i == BLOCK_SIZE) {
                // Write cache to MRAM output if full
                mutex_lock(mutex);
                uint32_t curr_pos = out_pos;
                out_pos += BLOCK_SIZE;
                mutex_unlock(mutex);
                mram_write(in_cache, (__mram_ptr void*) (out+curr_pos), BLOCK_SIZE*sizeof(key_ptr_t));
                out_i = 0;
            }
        }
    }

    if (out_i > 0) {
        // Write remaining cache to MRAM output
        mutex_lock(mutex);
        uint32_t curr_pos = out_pos;
        out_pos += out_i;
        mutex_unlock(mutex);
        mram_write(in_cache, (__mram_ptr void*) (out+curr_pos), out_i*sizeof(key_ptr_t));
    }
}

/*
    Main kernel for performing hash aggregation
*/
int group_kernel(aggr_arguments_t *input_args, aggr_results_t *result) {
    unsigned int tasklet_id = me();

    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
        table = mem_alloc(AGG_TABLE_SIZE*sizeof(key_ptr_t));
        memset(table, 0xff, AGG_TABLE_SIZE*sizeof(key_ptr_t));
    }
    // Barrier
    barrier_wait(&barrier);

    uint32_t input_size_dpu = input_args->size;

    // Address of the current processing block in MRAM
    uint32_t base_tasklet = tasklet_id * BLOCK_SIZE;
    key_ptr_t* mram_base_addr_A = (key_ptr_t*) input_args->in;
    key_ptr_t* mram_base_addr_B = (key_ptr_t*) input_args->out;

    // Initialize a local cache to store the MRAM block
    key_ptr_t *cache_proj = (key_ptr_t *) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    uint32_t rem_size = input_size_dpu;
    while (rem_size > 0) {
        uint32_t base = base_tasklet;
        for(; base < rem_size; base += BLOCK_SIZE * NR_TASKLETS) {
            uint32_t size = base + BLOCK_SIZE > rem_size ? rem_size % BLOCK_SIZE : BLOCK_SIZE;

            mram_read((__mram_ptr void*) (mram_base_addr_A + base), cache_proj, size*sizeof(key_ptr_t));

            uint32_t n_failed = hash_insert(cache_proj, size, input_args->aggr);

            barrier_wait(&barrier);
            // Write back the elements not inserted
            if (n_failed > 0) {
                mutex_lock(mutex);
                uint32_t curr_pos = wb_pos;
                wb_pos += n_failed;
                mutex_unlock(mutex);
                mram_write(cache_proj, (__mram_ptr void*) (mram_base_addr_A + curr_pos), n_failed*sizeof(key_ptr_t));
            }

        }

        // Sync the idle tasklets
        uint32_t size = (rem_size + NR_TASKLETS*BLOCK_SIZE - 1) / (NR_TASKLETS*BLOCK_SIZE);
        size *= NR_TASKLETS*BLOCK_SIZE;
        if (base < size) {
            barrier_wait(&barrier);
        }

        barrier_wait(&barrier);

        // Write the inserted elements to the output
        for (uint32_t bucket = tasklet_id; bucket < NR_BUCKETS; bucket += NR_TASKLETS) {
            output(cache_proj, table + bucket*BUCKET_SIZE, mram_base_addr_B);
        }

        rem_size = wb_pos;
        // Reset the table
        barrier_wait(&barrier);
        if (tasklet_id == 0) {
            memset(table, 0xff, AGG_TABLE_SIZE*sizeof(key_ptr_t));
        }
        wb_pos = 0;
        barrier_wait(&barrier);
    }

    barrier_wait(&barrier);
    result->t_count = out_pos;

    return 0;
}