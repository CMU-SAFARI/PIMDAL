/*
* Author: Manos Frouzakis
* Date: 1.12.2022
* 
*/

#include <defs.h>
#include <barrier.h>
#include <mram.h>
#include <alloc.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <mutex.h>
#include <handshake.h>

#include "hash_join.h"
#include "../hash/hash_steps.c"

// Number of elements in the table
uint32_t size_table;
// Size of filter
uint32_t size_filter;
// Size of partitions
uint32_t size_part;

// Multi tasklet cache to store data to MRAM
key_ptr32 *out_cache;
uint32_t out_index[NR_BUCKETS] = {0};
uint32_t size[NR_BUCKETS] = {0};
uint32_t indices[NR_BUCKETS] = {0};
uint32_t hist[NR_BUCKETS+1] = {0};

// Barrier defined in calling code
extern barrier_t barrier;

// Mutex defined in calling code
extern const mutex_id_t mutex;

int hash_kernel(hash_arguments_t *hash_args) {
    uint32_t tasklet_id = me();

    size_table = hash_args->table_size;
    key_ptr32* table = (key_ptr32*) hash_args->table_ptr;
    key_ptr32* buffer = (key_ptr32*) hash_args->in_ptr;

    if (tasklet_id == 0) {
        mem_reset();
        // Allocate shared cache
        out_cache = (key_ptr32*) mem_alloc(NR_BUCKETS * CACHE_SIZE * sizeof(key_ptr32));
        // Store partition size at the end of the region
        uint64_t nr_elements = hash_args->size;
        mram_write(&nr_elements, (__mram_ptr void*) (buffer + size_table - 1), sizeof(uint64_t));
    }
    barrier_wait(&barrier);
    key_ptr32* local_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));

    uint32_t shift = hash_args->shift;
    key_ptr32* in = buffer;
    key_ptr32* out = table;
    for (uint32_t table_size = size_table; table_size > 256; table_size /= NR_BUCKETS) {
        uint32_t nr_buckets = table_size / NR_BUCKETS >= 256 ? NR_BUCKETS : table_size / 256;
        shift -= __builtin_popcount(nr_buckets-1);
        //printf("Size: %u nr buckets: %u shift: %u\n", table_size / NR_BUCKETS, nr_buckets, shift);
        for (uint32_t subtable = 0; subtable < size_table; subtable += table_size) {
            barrier_wait(&barrier);
            uint64_t in_size;
            mram_read((__mram_ptr void*) (in+subtable+table_size-1), &in_size, sizeof(uint64_t));
            hash_step(in+subtable, in_size, out+subtable, table_size, local_cache,
                      (key_ptr32 (*)[CACHE_SIZE]) out_cache, nr_buckets, tasklet_id, shift);

            for (uint32_t bucket = 0; bucket < nr_buckets; bucket++) {
                if (bucket % NR_TASKLETS == tasklet_id) {
                    size[bucket] = 0;
                }
            }
        }

        key_ptr32* tmp_ptr = in;
        in = out;
        out = tmp_ptr;
    }

    barrier_wait(&barrier);
    if (tasklet_id == 0) {
        if (in == table) {
            memcpy((__mram_ptr void*) buffer, (__mram_ptr void*) table, size_table*sizeof(key_ptr32));
        }
        mem_reset();
    }
    barrier_wait(&barrier);

    uint32_t base_tasklet = tasklet_id * BLOCK_SIZE;
    uint32_t base_size = tasklet_id;
    local_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));
    key_ptr32* local_table = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));

    for (uint32_t base = base_tasklet; base < size_table; base += NR_TASKLETS*BLOCK_SIZE) {

        mram_read((__mram_ptr void*) (buffer+base), local_cache, BLOCK_SIZE*sizeof(key_ptr32));

        memset(local_table, 0xff, BLOCK_SIZE*sizeof(key_ptr32));

        uint64_t in_size = ((uint64_t*) local_cache)[255];
        //if (in_size > 255)
        //    printf("Split: %d Size: %lu\n", base, in_size);
        hash_phase_two(local_cache, in_size, local_table);

        mram_write(local_table, (__mram_ptr void*) (table+base), BLOCK_SIZE*sizeof(key_ptr32));

        base_size += NR_TASKLETS;
    }

    return 0;
}

int filter_kernel(filter_arguments_t *filter_args) {
    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptr32* input = (key_ptr32*) filter_args->in_ptr;
    uint8_t* filters = (uint8_t*) filter_args->filter_ptr;
    uint64_t* sizes_part = (uint64_t*) filter_args->part_sizes;
    uint32_t part_n = filter_args->part_n;

    key_ptr32* in_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));
    uint8_t* filter_cache = (uint8_t*) mem_alloc(FILTER_BYTES);

    uint32_t base_filter = tasklet_id*FILTER_BYTES;
    uint32_t partition = tasklet_id;

    for (uint32_t base = base_filter; base < part_n*FILTER_BYTES; base += NR_TASKLETS*FILTER_BYTES) {
        __dma_aligned uint64_t in_size[2];
        mram_read((__mram_ptr void*) (sizes_part + partition), &in_size, 2*sizeof(uint64_t));
        //printf("Start: %lu Size: %lu\n", in_size[0], in_size[1]);
        in_size[1] -= in_size[0];
        mram_read((__mram_ptr void*) (input + in_size[0]), in_cache, in_size[1]*sizeof(key_ptr32));

        memset(filter_cache, 0, FILTER_BYTES);

        insert_bloom_filter(in_cache, in_size[1], filter_cache);
        mram_write(filter_cache, (__mram_ptr void*) (filters+base), FILTER_BYTES);

        partition += NR_TASKLETS;
    }

    return 0;
}

int part_kernel(part_arguments_t *part_args) {
    uint32_t tasklet_id = me();

    key_ptr32* buffer = (key_ptr32*) part_args->in_ptr;
    key_ptr32* partitions = (key_ptr32*) part_args->part_ptr;
    uint64_t* sizes_part = (uint64_t*) part_args->part_sizes;

    if (tasklet_id == 0) {
        mem_reset();
        out_cache = (key_ptr32*) mem_alloc(NR_BUCKETS * CACHE_SIZE * sizeof(key_ptr32));
        uint64_t start = 0;
        mram_write(&start, (__mram_ptr void*) (sizes_part), sizeof(uint64_t));
        uint64_t nr_elements = part_args->size;
        mram_write(&nr_elements, (__mram_ptr void*) (sizes_part+part_args->part_n), sizeof(uint64_t));
    }
    barrier_wait(&barrier);

    key_ptr32* local_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));
    //key_ptr32* local_table = (key_ptr32*) mem_alloc(4*TABLE_2_SIZE*sizeof(key_ptr32));

    uint32_t shift = part_args->shift;
    key_ptr32* in = buffer;
    key_ptr32* out = partitions;
    for (uint32_t partition_size = part_args->part_n; partition_size > 1; partition_size /= NR_BUCKETS) {
        uint32_t nr_buckets = partition_size >= NR_BUCKETS ? NR_BUCKETS : partition_size;
        shift -= __builtin_popcount(nr_buckets-1);

        for (uint32_t partition = 0; partition < part_args->part_n; partition += partition_size) {
            //printf("Part size: %u\n", partition_size);
            // Reset histogram
            for (uint32_t bucket = 0; bucket < nr_buckets+1; bucket++) {
                if (bucket % NR_TASKLETS == tasklet_id) {
                    hist[bucket] = 0;
                }
            }

            uint64_t in_start;
            mram_read((__mram_ptr void*) (sizes_part+partition), &in_start, sizeof(uint64_t));
            uint64_t in_size;
            mram_read((__mram_ptr void*) (sizes_part+partition+partition_size), &in_size, sizeof(uint64_t));
            in_size -= in_start;
            // if (tasklet_id == 0) {
            //     printf("%u start: %lu %u size: %lu\n", partition, in_start, partition+partition_size, in_size);
            //     printf("In: %p %p\n", in+in_start, out+in_start);
            // }
            barrier_wait(&barrier);
            
            build_histogram(tasklet_id, hist, local_cache, in+in_start, in_size, nr_buckets, shift);
            barrier_wait(&barrier);

            prefix_sum(tasklet_id, hist, nr_buckets);
            barrier_wait(&barrier);

            for (uint32_t bucket = 0; bucket < nr_buckets; bucket++) {
                if (bucket % NR_TASKLETS == tasklet_id) {
                    uint32_t part_size = partition + (bucket+1)*partition_size/nr_buckets;
                    uint64_t out_start = in_start + hist[bucket+1];
                    mram_write(&out_start, (__mram_ptr void*) (sizes_part+part_size), sizeof(uint64_t));
                    //printf("%u Size: %lu %lu\n", part_size, in_start, out_start);
                }
            }
            barrier_wait(&barrier);

            write_output(tasklet_id, hist, local_cache, in+in_start, (key_ptr32 (*)[CACHE_SIZE]) out_cache,
                         out+in_start, in_size, nr_buckets, shift);
        }

        key_ptr32* tmp_ptr = in;
        in = out;
        out = tmp_ptr;
    }

    barrier_wait(&barrier);

    if (tasklet_id == 0) {
        if (out == partitions) {
            memcpy((__mram_ptr void*) partitions, (__mram_ptr void*) buffer, part_args->size*sizeof(key_ptr32));
        }
    }

    return 0;
}

int match_kernel(match_arguments_t *match_args, merge_results_t *match_res) {

    uint32_t tasklet_id = me();

    uint8_t* bloom_filter = (uint8_t*) match_args->filter_ptr;
    key_ptr32* partitions = (key_ptr32*) match_args->part_ptr;
    key_ptr32* matches = (key_ptr32*) match_args->out_ptr;
    uint64_t* sizes_match = (uint64_t*) match_args->match_sizes;
    uint64_t* sizes_part = (uint64_t*) match_args->part_sizes;

    if (tasklet_id == 0) {
        mem_reset();
        hist[0] = 0;
        memset((__mram_ptr void*) sizes_match, 0, (match_args->part_n + 1)*sizeof(uint64_t));
    }
    barrier_wait(&barrier);

    uint8_t* filter_cache = (uint8_t*) mem_alloc(FILTER_BYTES);
    key_ptr32* local_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));

    uint32_t base_tasklet = tasklet_id * BLOCK_SIZE;
    uint32_t partition = 0;
    __dma_aligned uint64_t in_end;
    mram_read((__mram_ptr void*) (sizes_part+1), &in_end, sizeof(uint64_t));
    //printf("%u Base: %u, end: %lu\n", tasklet_id, base_tasklet, in_end);

    uint32_t size = (match_args->size + NR_TASKLETS*BLOCK_SIZE - 1) / (NR_TASKLETS*BLOCK_SIZE) * NR_TASKLETS*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < size; base += BLOCK_SIZE*NR_TASKLETS) {
        // Read the input for this iteration
        mram_read((__mram_ptr void*) (partitions + base), local_cache, BLOCK_SIZE*sizeof(key_ptr32));

        uint32_t hits_iter = 0;
        uint32_t n = 0;
        for (uint32_t in_offset = 0; in_offset < BLOCK_SIZE; in_offset += n) {
            if (partition > match_args->part_n) {
                break;
            }
            if (base + in_offset >= in_end) {
                // Find the the filter corresponding to the partition
                partition++;
                mram_read((__mram_ptr void*) (sizes_part+partition+1), &in_end, sizeof(uint64_t));
                //printf("%u Base: %u, end: %lu\n", tasklet_id, base+in_offset, in_end);
                n = 0;
            }
            else {
                mram_read((__mram_ptr void*) (bloom_filter+partition*FILTER_BYTES), filter_cache, FILTER_BYTES*sizeof(uint8_t));
                if (BLOCK_SIZE - in_offset > in_end - base - in_offset) {
                    n = in_end - base - in_offset;
                    uint32_t hits = probe_bloom_filter(local_cache+in_offset, local_cache+hits_iter, n, filter_cache);
                    //printf("%u New hits: %u n: %u partition: %u\n", tasklet_id, hits, n, partition);
                    // Move new hits to front
                    //memcpy(local_cache+hits_iter, local_cache+in_offset, hits*sizeof(key_ptr32));
                    hits_iter += hits;

                    // Add new hits to total count
                    mutex_lock(mutex);
                    uint64_t hits_part;
                    mram_read((__mram_ptr void*) (sizes_match+partition+1), &hits_part, sizeof(uint64_t));
                    hits_part += hits;
                    mram_write(&hits_part, (__mram_ptr void*) (sizes_match+partition+1), sizeof(uint64_t));
                    mutex_unlock(mutex);
                }
                else {
                    n = BLOCK_SIZE - in_offset;
                    uint32_t hits = probe_bloom_filter(local_cache+in_offset, local_cache+hits_iter, n, filter_cache);
                    //printf("%u New hits: %u n: %u partition: %u\n", tasklet_id, hits, n, partition);
                    //memcpy(local_cache+hits_iter, local_cache+in_offset, hits*sizeof(key_ptr32));
                    hits_iter += hits;

                    // Add new hits to total count
                    mutex_lock(mutex);
                    uint64_t hits_part;
                    mram_read((__mram_ptr void*) (sizes_match+partition+1), &hits_part, sizeof(uint64_t));
                    hits_part += hits;
                    mram_write(&hits_part, (__mram_ptr void*) (sizes_match+partition+1), sizeof(uint64_t));
                    mutex_unlock(mutex);
                }
            }
        }

        uint32_t out_off = 0;
        if (tasklet_id > 0) {
            handshake_wait_for(tasklet_id-1);
        }
        out_off = hist[tasklet_id];
        if (tasklet_id < NR_TASKLETS-1) {
            hist[tasklet_id+1] = hist[tasklet_id] + hits_iter;
            handshake_notify();
        }
        else {
            hist[0] = hist[tasklet_id] + hits_iter;
        }
    
        if (hits_iter > 0) {
            mram_write(local_cache, (__mram_ptr void*) (matches+out_off), hits_iter*sizeof(key_ptr32));
        }
        barrier_wait(&barrier);
    }
    //printf("%u nr elements: %u\n", tasklet_id, hist[tasklet_id]);
    if (tasklet_id == 0) {
        match_res->out_n = hist[tasklet_id];
    }

    return 0;
}

int merge_kernel(merge_arguments_t *join_args, merge_results_t *join_res) {
    
    uint32_t tasklet_id = me();

    key_ptr32* table = (key_ptr32*) join_args->table_ptr;
    key_ptr32* partitions = (key_ptr32*) join_args->in_ptr;
    key_ptr32* joined = (key_ptr32*) join_args->out_ptr;
    uint64_t* sizes_part = (uint64_t*) join_args->part_sizes;

    if (tasklet_id == 0) {
        mem_reset();
        hist[0] = 0;
    }
    barrier_wait(&barrier);

    key_ptr32* table_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));
    key_ptr32* local_cache = (key_ptr32*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr32));

    uint32_t base_tasklet = tasklet_id * BLOCK_SIZE;
    uint32_t partition = 0;
    __dma_aligned uint64_t in_end;
    mram_read((__mram_ptr void*) (sizes_part+1), &in_end, sizeof(uint64_t));

    uint32_t size = (join_args->size + NR_TASKLETS*BLOCK_SIZE - 1) / (NR_TASKLETS*BLOCK_SIZE);
    size *= NR_TASKLETS*BLOCK_SIZE;
    //printf("Size: %u Part: %u\n", size, join_args->part_n);
    for (uint32_t base = base_tasklet; base < size; base += BLOCK_SIZE*NR_TASKLETS) {
        // Read the input for this iteration
        mram_read((__mram_ptr void*) (partitions + base), local_cache, BLOCK_SIZE*sizeof(key_ptr32));

        uint32_t hits_iter = 0;
        uint32_t n = 0;
        for (uint32_t in_offset = 0; in_offset < BLOCK_SIZE; in_offset += n) {
            if (partition > join_args->part_n) {
                break;
            }
            if (base + in_offset >= in_end) {
                // Find the the filter corresponding to the partition
                partition++;
                mram_read((__mram_ptr void*) (sizes_part+partition+1), &in_end, sizeof(uint64_t));
                //printf("%u Base: %u, end: %lu\n", tasklet_id, base+in_offset, in_end);
                n = 0;
            }
            else {
                mram_read((__mram_ptr void*) (table+partition*BLOCK_SIZE), table_cache, BLOCK_SIZE*sizeof(key_ptr32));
                if (BLOCK_SIZE - in_offset > in_end - base - in_offset) {
                    n = in_end - base - in_offset;
                    uint32_t hits = probe_table(local_cache+in_offset, local_cache+hits_iter, table_cache, n);
                    //printf("%u New hits: %u n: %u partition: %u\n", tasklet_id, hits, n, partition);
                    // Move new hits to front
                    //memcpy(local_cache+hits_iter, local_cache+in_offset, hits*sizeof(key_ptr32));
                    hits_iter += hits;
                }
                else {
                    n = BLOCK_SIZE - in_offset;
                    uint32_t hits = probe_table(local_cache+in_offset, local_cache+hits_iter, table_cache, n);
                    //printf("%u New hits: %u n: %u partition: %u\n", tasklet_id, hits, n, partition);
                    //memcpy(local_cache+hits_iter, local_cache+in_offset, hits*sizeof(key_ptr32));
                    hits_iter += hits;
                }
            }
        }

        uint32_t out_off = 0;
        if (tasklet_id > 0) {
            handshake_wait_for(tasklet_id-1);
        }
        out_off = hist[tasklet_id];
        if (tasklet_id < NR_TASKLETS-1) {
            hist[tasklet_id+1] = hist[tasklet_id] + hits_iter;
            handshake_notify();
        }
        else {
            hist[0] = hist[tasklet_id] + hits_iter;
        }
    
        if (hits_iter > 0) {
            mram_write(local_cache, (__mram_ptr void*) (joined+out_off), hits_iter*sizeof(key_ptr32));
        }
        barrier_wait(&barrier);
    }

    if (tasklet_id == NR_TASKLETS-1) {
        join_res->out_n = hist[0];
    }

    return 0;
}