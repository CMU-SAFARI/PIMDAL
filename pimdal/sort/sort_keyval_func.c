#include <mram.h>
#include <alloc.h>
#include <stdint.h>

#include "datatype.h"

#ifndef SORT_BLOCK_SIZE
#define SORT_BLOCK_SIZE 64
#endif
#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif
#ifndef NR_SPLITS
#define NR_SPLITS 2048
#endif

uint32_t part_step(key_ptr_t *left_cache, key_ptr_t *right_cache, uint64_t n,
                         int64_t *i, int64_t *j, key_ptr_t pivot) {

    int64_t left = *i % SORT_BLOCK_SIZE;
    int64_t right = SORT_BLOCK_SIZE - 1 - (n - *j) % SORT_BLOCK_SIZE;
    while (*i < *j) {
        if (left_cache[left].key > pivot.key) {
            if (right_cache[right].key <= pivot.key) {
                key_ptr_t tmp = left_cache[left];
                left_cache[left] = right_cache[right];
                right_cache[right] = tmp;
                (*i)++;
                (*j)--;
                left++;
                right--;
            }
            else {
                (*j)--;
                right--;
            }
        }
        else {
            (*i)++;
            left++;
        }

        if (left == SORT_BLOCK_SIZE && right == -1) {
            return 1;
        }
        else if (left == SORT_BLOCK_SIZE) {
            return 2;
        }
        if (right == -1) {
            return 3;
        }
    }

    return 0;
}

uint32_t quick_sort_step(key_ptr_t *left_cache, key_ptr_t *right_cache, uint64_t n,
                         int64_t *i, int64_t *j, key_ptr_t pivot) {

    int64_t left = *i % SORT_BLOCK_SIZE;
    int64_t right = SORT_BLOCK_SIZE - 1 - (n - *j) % SORT_BLOCK_SIZE;
    while (*i < *j) {
        if (left_cache[left].key >= pivot.key) {
            if (right_cache[right].key <= pivot.key) {
                key_ptr_t tmp = left_cache[left];
                left_cache[left] = right_cache[right];
                right_cache[right] = tmp;
                (*i)++;
                (*j)--;
                left++;
                right--;
            }
            else {
                (*j)--;
                right--;
            }
        }
        else {
            (*i)++;
            left++;
        }

        if (left == SORT_BLOCK_SIZE && right == -1) {
            return 1;
        }
        else if (left == SORT_BLOCK_SIZE) {
            return 2;
        }
        if (right == -1) {
            return 3;
        }
    }

    return 0;
}

static void selection_sort(key_ptr_t *input, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        key_ptr_t min = input[i];
        int32_t min_i = i;

        for (int32_t j = i+1; j < n; j++) {
            if(input[j].key < min.key) {
                min = input[j];
                min_i = j;
            }
        }

        if (min_i != i) {
            input[min_i] = input[i];
            input[i] = min;
        }
    }
}

static uint32_t quick_sort(key_ptr_t *input, int64_t n) {

    uint32_t pivot = input[n/2].key;

    int32_t i = 0;
    int32_t j = n-1;

    while (i <= j) {
        while (input[i].key < pivot) {
            i++;
        }

        while (input[j].key > pivot) {
            j--;
        }
        
        if (i <= j) {
            key_ptr_t tmp = input[i];
            input[i] = input[j];
            input[j] = tmp;
            i++;
            j--;
        }
    }

    //printf("pivot: %lu\n", pivot);
    //printf("i: %d j: %d data: %lu %lu\n", i, j, input[i], input[j]);
    return i;
}

/*
*/
void sort_wram(key_ptr_t *local_cache, uint64_t n) {
    //Contain the boundaries of the simulated recursion levels
    //With 32 indexes, 2^32 elements can be ordered
    const int32_t max_levels = 12;
    uint32_t level_start[max_levels];
    uint32_t level_end[max_levels];

    uint32_t local_start, local_end;  //Indexes of the elements inside each simulated recursion
    int32_t i = 0;  //Simulated recursion level (can be negative)

    level_start[0] = 0;  //First simulated recursion spans across all assigned section of the sample
    level_end[0] = n;

    while (i >= 0) {  ///While there are still some levels to handle

        local_start = level_start[i];  //local start at level i
        local_end = level_end[i];  //local end at level i

        if (local_start < local_end) {

            uint32_t size = local_end - local_start;
            if (size < 10) {
                selection_sort(local_cache+local_start, size);

                i--;
            }
            else {
                //Partition the the array
                uint32_t p = quick_sort(local_cache+local_start, size);
                //printf("%d %u - %u %u\n", i, local_start, local_end, p);

                //The next level will sort the left section
                level_start[i+1] = local_start;
                level_end[i+1] = local_start + p;

                //Overwritten the current level. It will sort the right section
                level_start[i] = local_start + p;

                i++;  //Increase one level

                //If this level(after i++) is bigger than the previous one, swap them (execute before the smaller one. Tail "recursion")
                //It is guaranteed that 2^(max_levels) elements can be ordered
                if(level_end[i] - level_start[i] > level_end[i-1] - level_start[i-1]){
                    uint32_t temp = level_start[i];
                    level_start[i] = level_start[i-1];
                    level_start[i-1] = temp;

                    temp = level_end[i];
                    level_end[i] = level_end[i-1];
                    level_end[i-1] = temp;
                }
            }

        } else {
            //If the level does not need to be ordered, order the level i-1
            i--;
        }
    }
}

uint32_t sort_blocks(key_ptr_t *in, key_ptr_t *out, uint32_t n, key_ptr_t *left_cache,
                     key_ptr_t *right_cache, key_ptr_t pivot, uint32_t part){

    uint32_t left_i = 0;
    uint32_t right_i = n - SORT_BLOCK_SIZE;
    uint32_t status = 0;

    int64_t i = 0;
    int64_t j = n;

    mram_read((__mram_ptr void*)(in), left_cache, SORT_BLOCK_SIZE*sizeof(key_ptr_t));
    mram_read((__mram_ptr void*)(in + right_i), right_cache, SORT_BLOCK_SIZE*sizeof(key_ptr_t));
    do {
        if (part) {
            status = part_step(left_cache, right_cache, n, &i, &j, pivot);
        }
        else {
            status = quick_sort_step(left_cache, right_cache, n, &i, &j, pivot);
        }

        if (status == 1 || status == 2) {
            mram_write(left_cache, (__mram_ptr void*) (out + left_i), SORT_BLOCK_SIZE*sizeof(key_ptr_t));
            left_i += SORT_BLOCK_SIZE;
            mram_read((__mram_ptr void*)(in + left_i), left_cache, SORT_BLOCK_SIZE*sizeof(key_ptr_t));          
        }
        if (status == 1 || status == 3) {
            mram_write(right_cache, (__mram_ptr void*) (out + right_i), SORT_BLOCK_SIZE*sizeof(key_ptr_t));
            right_i -= SORT_BLOCK_SIZE;
            mram_read((__mram_ptr void*)(in + right_i), right_cache, SORT_BLOCK_SIZE*sizeof(key_ptr_t));
        }
    } while (status != 0);

    uint32_t nr_left = i % SORT_BLOCK_SIZE;
    uint32_t nr_right = (n - j)%SORT_BLOCK_SIZE;
    if (nr_left > 0) {
        mram_write(left_cache, (__mram_ptr void*) (out + left_i), nr_left*sizeof(key_ptr_t));
    }
    if (nr_right > 0) {
        mram_write(right_cache + SORT_BLOCK_SIZE - nr_right, (__mram_ptr void*) (out + right_i + SORT_BLOCK_SIZE - nr_right), nr_right*sizeof(key_ptr_t));
    }

    return i;
}

/*
Fully sort an array in MRAM using qucksort with random pivot selection.
*/
void sort_full(key_ptr_t *in, key_ptr_t *out, uint32_t n, key_ptr_t *local_cache) {
    //Contain the boundaries of the simulated recursion levels
    //With 32 indexes, 2^32 elements can be ordered
    const int32_t max_levels = 32;
    uint32_t level_start[max_levels];
    uint32_t level_end[max_levels];

    uint32_t local_start, local_end;  //Indexes of the elements inside each simulated recursion
    int32_t i = 0;  //Simulated recursion level (can be negative)

    level_start[0] = 0;  //First simulated recursion spans across all assigned section of the sample
    level_end[0] = n;

    uint32_t rand = 1;
    __dma_aligned key_ptr_t pivots[5];

    while (i >= 0) {  ///While there are still some levels to handle

        local_start = level_start[i];  //local start at level i
        local_end = level_end[i];  //local end at level i
        //printf("%d %u - %u\n", i, local_start, local_end);

        if (local_start < local_end) {
            //Partition the the array
            uint32_t size = local_end - local_start;
            if (size <= 2*SORT_BLOCK_SIZE) {
                //printf("%d: %u - %u\n", i, local_start, local_end);
                mram_read((__mram_ptr void*) (in+local_start), local_cache, size*sizeof(key_ptr_t));
                sort_wram(local_cache, size);
                mram_write(local_cache, (__mram_ptr void*) (out+local_start), size*sizeof(key_ptr_t));

                // Current level has been sorted
                level_start[i] = local_end;
                i--;
            }
            else {
                mram_read((__mram_ptr void*) (in+local_start+rand), pivots, 5*sizeof(key_ptr_t));
                selection_sort(pivots, 5);
                uint32_t p = sort_blocks(in + local_start, out + local_start, size, local_cache, local_cache+SORT_BLOCK_SIZE, pivots[2], 0);
                //printf("%d %u - %u %u %lu\n", i, local_start, local_end, p, pivots[2]);

                //The next level will sort the left section
                level_start[i+1] = local_start;
                level_end[i+1] = local_start + p;

                //Overwrite the current level. It will sort the right section
                level_start[i] = local_start + p;

                i++;  //Increase one level

                //If this level(after i++) is bigger than the previous one, swap them (execute before the smaller one. Tail "recursion")
                //It is guaranteed that 2^(max_levels) elements can be ordered
                if(level_end[i] - level_start[i] > level_end[i-1] - level_start[i-1]){
                    uint32_t temp = level_start[i];
                    level_start[i] = level_start[i-1];
                    level_start[i-1] = temp;

                    temp = level_end[i];
                    level_end[i] = level_end[i-1];
                    level_end[i-1] = temp;
                }
            }

            rand = (5*rand + 1)%SORT_BLOCK_SIZE;
        } else {
            //If the level does not need to be ordered, order the level i-1
            i--;
        }
    }
}

void reorder(key_ptr_t *input, key_ptr_t *output, key_ptr_t* local_cache, uint32_t tasklet_id,
             uint64_t indices_loc[NR_SPLITS][NR_TASKLETS], uint64_t indices_off[NR_SPLITS][NR_TASKLETS],
             uint32_t nr_splits) {

    // The offset to load the elements
    uint64_t offset_in = 0;
    // The offset to store the elements
    uint64_t offset_glob = 0;
    for (uint32_t i = 0; i < nr_splits; i++) {
        uint64_t offset = 0;
        uint64_t length = 0;
        if (tasklet_id > 0) {
            mram_read((__mram_ptr void*) &indices_off[i][tasklet_id-1], &offset, sizeof(uint64_t));
        }
        mram_read((__mram_ptr void*) &indices_loc[i][tasklet_id], &length, sizeof(uint64_t));

        uint64_t offset_tot = offset + offset_glob;

        for (uint32_t base = 0; base + 2*SORT_BLOCK_SIZE <= length; base += 2*SORT_BLOCK_SIZE) {
            mram_read((__mram_ptr void*) (input + offset_in), local_cache, 2*SORT_BLOCK_SIZE*sizeof(key_ptr_t));

            mram_write(local_cache, (__mram_ptr void*) (output + offset_tot), 2*SORT_BLOCK_SIZE*sizeof(key_ptr_t));
            offset_in += 2*SORT_BLOCK_SIZE;
            offset_tot += 2*SORT_BLOCK_SIZE;
        }
        uint64_t rem_size = length % (2*SORT_BLOCK_SIZE);
        if (rem_size > 0) {
            mram_read((__mram_ptr void*) (input + offset_in), local_cache, rem_size*sizeof(key_ptr_t));

            mram_write(local_cache, (__mram_ptr void*) (output + offset_tot), rem_size*sizeof(key_ptr_t));
        }

        offset_in += rem_size;
        
        uint64_t offset_glob_tmp;
        mram_read((__mram_ptr void*) &indices_off[i][NR_TASKLETS-1], &offset_glob_tmp, sizeof(uint64_t));
        offset_glob += offset_glob_tmp;
    }
}