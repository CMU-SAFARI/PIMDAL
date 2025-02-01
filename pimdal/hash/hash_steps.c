#include <barrier.h>
#include <mutex_pool.h>

#include "datatype.h"
#include "linear_hash.c"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif
#define NR_BUCKETS 32
#define CACHE_SIZE (2048/NR_BUCKETS)
#define FILTER_SIZE 16384
#define FILTER_BYTES (FILTER_SIZE/8)

extern uint32_t out_index[NR_BUCKETS];
extern uint32_t size[NR_BUCKETS];
extern uint32_t indices[NR_BUCKETS];
extern uint32_t hist[NR_BUCKETS+1];

extern const mutex_id_t mutex;
extern barrier_t barrier;
extern struct mutex_pool mutexes;

/*
    @param out the cache to write to MRAM
    @param n the number of elements to write
    @param glob_table MRAM pointer to write to
    @param bucket which bucket to write

Writes a bucket in the shared cache to MRAM
*/
void insert_bucket(key_ptr32 out[NR_BUCKETS][CACHE_SIZE], uint32_t n,
                   key_ptr32 **glob_table, uint32_t bucket) {

    uint32_t offset = indices[bucket];
    indices[bucket] += n;
    mram_write(out[bucket], (__mram_ptr void*) (glob_table[bucket] + offset), n*sizeof(key_ptr32));
    //printf("%u Out: %u Size: %u\n", bucket, offset, n);
}

/*
    @param in the input data
    @param n number of input elements
    @param output_cache the shared WRAM cache
    @param size how many elements are in each bucket
    @param glob_table the MRAM location to store the partitions to
    @param nr_buckets the number of buckets used in a partitioning step
    @param shift shift used in hash function in the current step
*/
void hash_phase_one(key_ptr32* in, uint32_t n, key_ptr32 output_cache[NR_BUCKETS][CACHE_SIZE], uint32_t *size,
                    key_ptr32 **glob_table, uint32_t nr_buckets, uint32_t shift) {

    for (uint32_t i = 0; i < n; i++) {
        uint32_t hash = hash0(in[i].key);
        hash = (hash >> shift) & (nr_buckets-1);
        //if (i == 0)
        //    printf("Key: %x Hash: %u shift: %u\n", in[i].key, hash, shift);

        mutex_pool_lock(&mutexes, hash);
        //printf("Hash: %u Offset: %u\n", hash, size[hash]);
        output_cache[hash][size[hash]] = in[i];
        // Copy bucket in cache to MRAM if full
        if (size[hash] == CACHE_SIZE - 1) {
            insert_bucket(output_cache, CACHE_SIZE, glob_table, hash);
            size[hash] = 0;
        }
        else {
            size[hash]++;
        }
        mutex_pool_unlock(&mutexes, hash);
    }
}

/*
    @param in input elements in MRAM
    @param in_size number of elements
    @param out output in MRAM
    @param table_size total size of the table in the current step
    @param local_cache WRAM cache used execution
    @param output_cache shared WRAM cache used for writing elements to the output
    @param nr_buckets number of buckets used in this iteration
    @param tasklet_id id of the calling tasklet
    @param shift which bits of the hash value to use in this iteration

    One iteration of partitioning the input into nr_buckets partitions.
*/
void hash_step(key_ptr32* in, uint32_t in_size, key_ptr32* out, uint32_t table_size,
               key_ptr32* local_cache, key_ptr32 output_cache[NR_BUCKETS][CACHE_SIZE],
               uint32_t nr_buckets, uint32_t tasklet_id, uint32_t shift) {

    // Store the indices of the buckets in array to save calculations
    key_ptr32 *table_split[NR_BUCKETS];
    for (uint32_t bucket_offset = 0; bucket_offset < nr_buckets; bucket_offset++) {
        table_split[bucket_offset] = (key_ptr32*) (out + bucket_offset * (table_size/nr_buckets));
    }

    // Perform partitioning step
    uint32_t base_tasklet = tasklet_id * BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < in_size; base += NR_TASKLETS*BLOCK_SIZE) {
        uint32_t batch_size = base + BLOCK_SIZE < in_size ? BLOCK_SIZE : in_size - base;

        mram_read((__mram_ptr void*) (in+base), local_cache, batch_size*sizeof(key_ptr32));

        hash_phase_one(local_cache, batch_size, output_cache, size,
                       table_split, nr_buckets, shift);
    }

    barrier_wait(&barrier);

    // Write remaining buckets to MRAM
    for (uint32_t bucket = 0; bucket < nr_buckets; bucket++) {
        if (bucket % NR_TASKLETS == tasklet_id) {
            if (size[bucket] > 0) {
                insert_bucket(output_cache, size[bucket], table_split, bucket);
            }
        }
    }

    barrier_wait(&barrier);

    // Store size of bucket at the end of its location
    for (uint32_t bucket = 0; bucket < nr_buckets; bucket++) {
        if (bucket % NR_TASKLETS == tasklet_id) {
            uint64_t len = indices[bucket];
            mram_write(&len, (__mram_ptr void*) (out + (bucket+1)*(table_size/nr_buckets) - 1),
                       sizeof(uint64_t));
            indices[bucket] = 0;
            //printf("Bucket size: %lu\n", len);
        }
    }
}

/*
    @param in WRAM cache of input elements
    @param n number of elements
    @out bloom filter in WRAM

    Create bloom filter indicating membership of the input elements.
*/
void insert_bloom_filter(key_ptr32* in, uint32_t n, uint8_t* out) {
    for (uint32_t filter = 0; filter < 4; filter++) {
        for (uint32_t i = 0; i < n; i++) {
            uint32_t bit = hash_func[filter](in[i].key);
            bit = bit % FILTER_SIZE;
            out[bit / 8] = out[bit / 8] | ((uint8_t) 1 << (bit % 8));
        }
    }
}

/*
    @param in WRAM cache of input elements
    @param out elements that hit in the bloom filter
    @param n number of input elements
    @param filters bloom filters

    Probe the bloom filter with input elements to find the wanted elements.
*/
uint32_t probe_bloom_filter(key_ptr32* in, key_ptr32* out, uint32_t n, uint8_t* filters) {
    uint32_t selected = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t hit = 1;
        for (uint32_t filter = 0; filter < 4; filter++) {
            uint32_t bit = hash_func[filter](in[i].key);
            bit = bit % FILTER_SIZE;

            if ((filters[bit / 8] & ((uint8_t) 1 << (bit % 8))) == 0) {
                hit = 0;
                break;
            }
        }
        if (hit == 1) {
            out[selected] = in[i];
            selected++;
        }
        //else {
            //printf("Not found: %u\n", in[i].identifier);
        //}
    }

    return selected;
}

/*
    @param tasklet_id id of the calling tasklet
    @param histogram histogram counting the number of elements in each bucket
    @param input_cache WRAM cache used for the input elements
    @param input input elements in MRAM
    @param n number of input elements
    @param nr_buckets number of buckets used in this iteration
    @param shift which bits of the hash value to use in this iteration

    Build a histogram of the hash values for this iteration to calculate the output
    offset of the partitions.
*/
void build_histogram(uint32_t tasklet_id, uint32_t* histogram, key_ptr32 *input_cache,
                     key_ptr32 *input, uint32_t n, uint32_t nr_buckets, uint32_t shift) {

    uint32_t hist_loc[NR_BUCKETS] = {0};
    for (unsigned int block_offset = tasklet_id * BLOCK_SIZE;
        block_offset < n; block_offset += BLOCK_SIZE * NR_TASKLETS) {
        mram_read((__mram_ptr void*) (input + block_offset), input_cache, BLOCK_SIZE*sizeof(key_ptr32));

        uint32_t size = block_offset + BLOCK_SIZE < n ? BLOCK_SIZE : n - block_offset;
        for (unsigned int i = 0; i < size; ++i) {
            key_ptr32 item = input_cache[i];
            uint32_t bucket = hash0(item.key);
            bucket = (bucket >> shift) & (nr_buckets-1);
            hist_loc[bucket]++;
        }
    }

    for (uint32_t bucket = 0; bucket < nr_buckets; bucket++) {
        mutex_pool_lock(&mutexes, bucket);
        histogram[bucket+1] += hist_loc[bucket];
        mutex_pool_unlock(&mutexes, bucket);
    }
}

/*
    @param tasklet_id id of the calling tasklet
    @param created histogram of the hash values
    @param n number of partitions shown in the histogram

    Calculate a prefix sum of the histogram to get partition offsets.
*/
void prefix_sum(uint32_t tasklet_id, uint32_t *histogram, uint32_t n) {
    if (tasklet_id == 0) {
        for (uint32_t i = 0; i < n; i++) {
            histogram[i+1] += histogram[i];
        }
    }
}

/*
    @param tasklet_id id of the calling tasklet
    @param histogram offsets of the partitions calculated from histogram
    @param input_cache WRAM cache used for the input elements
    @param input input elements in MRAM
    @param output_cache shared WRAM cache to write elements to output
    @param output partitions output in MRAM
    @param n number of input elements
    @param nr_buckets number of buckets used in this iteration
    @param shift which bits of the hash value to use in this iteration

    Partition the elements using the offsets previously calculated with the histogram.
*/
void write_output(uint32_t tasklet_id, uint32_t* histogram, key_ptr32 *input_cache,
                  key_ptr32 *input, key_ptr32 output_cache[NR_BUCKETS][CACHE_SIZE], key_ptr32* output,
                 uint32_t n, uint32_t nr_buckets, uint32_t shift) {

    for (uint32_t block_offset = tasklet_id * BLOCK_SIZE;
        block_offset < n; block_offset += BLOCK_SIZE * NR_TASKLETS) {
        mram_read((__mram_ptr void*) (input + block_offset), input_cache, BLOCK_SIZE*sizeof(key_ptr32));

        uint32_t size = block_offset + BLOCK_SIZE < n ? BLOCK_SIZE : n - block_offset;
        for (unsigned int i = 0; i < size; ++i) {
            key_ptr32 item = input_cache[i];
            uint32_t bucket = hash0(item.key);
            bucket = (bucket >> shift) & (nr_buckets-1);
            //uint32_t bucket = (item.identifier >> shift) % NR_BUCKETS;

            mutex_pool_lock(&mutexes, bucket);
            output_cache[bucket][out_index[bucket]] = item;
            if (out_index[bucket] == CACHE_SIZE-1) {
                uint32_t out_offset = histogram[bucket];
                //printf("write %u: %p size: %u\n", bucket, output+out_offset, BLOCK_SIZE);
                mram_write(&output_cache[bucket][0], (__mram_ptr void*) (output+out_offset),
                           CACHE_SIZE*sizeof(key_ptr32));
                histogram[bucket] += CACHE_SIZE;
                out_index[bucket] = 0;
            }
            else {
                out_index[bucket]++;
            }
            mutex_pool_unlock(&mutexes, bucket);
        }
    }

    barrier_wait(&barrier);

    for (uint32_t bucket = 0; bucket < nr_buckets; bucket++) {
        if (bucket % NR_TASKLETS == tasklet_id && out_index[bucket] > 0) {
            uint32_t out_offset = histogram[bucket];
            //printf("write %u: %p size: %u\n", tasklet_id, output+out_offset, out_index[tasklet_id]);
            mram_write(&output_cache[bucket][0], (__mram_ptr void*) (output+out_offset),
                       out_index[bucket]*sizeof(key_ptr32));

            out_index[bucket] = 0;
        }
    }
}
