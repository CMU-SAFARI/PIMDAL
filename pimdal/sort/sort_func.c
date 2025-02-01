#include <mram.h>
#include <alloc.h>
#include <stdint.h>

#include "../support/datatype.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif
#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif
#ifndef NR_SPLITS
#define NR_SPLITS 1024
#endif

uint32_t quick_sort_step(T *left_cache, T *right_cache, uint64_t n,
                         int64_t *i, int64_t *j, T pivot) {

    int64_t left = *i % BLOCK_SIZE;
    int64_t right = BLOCK_SIZE - 1 - (n - *j) % BLOCK_SIZE;
    while (*i < *j) {
        if (left_cache[left] > pivot) {
            if (right_cache[right] <= pivot) {
                T tmp = left_cache[left];
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

        if (left == BLOCK_SIZE && right == -1) {
            return 1;
        }
        else if (left == BLOCK_SIZE) {
            return 2;
        }
        if (right == -1) {
            return 3;
        }
    }

    return 0;
}

static void selection_sort(T *input, int32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        T min = input[i];
        uint32_t min_i = i;

        for (uint32_t j = i+1; j < n; j++) {
            if(input[j] < min) {
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

static void quick_sort(T *input, int32_t n) {
    if (n < 10) {
        selection_sort(input, n);
        return;
    }

    T pivot = input[n/2];

    int32_t i = 0;
    int32_t j = n-1;

    while (i <= j) {
        while (input[i] < pivot) {
            i++;
        }

        while (input[j] > pivot) {
            j--;
        }
        
        if (i <= j) {
            T tmp = input[i];
            input[i] = input[j];
            input[j] = tmp;
            i++;
            j--;
        }
    }

    if (j > 0) {
        quick_sort(input, j+1);
    }
    if (n-i > 1){
        quick_sort(input+i, n-i);
    }
}

uint32_t sort_blocks(T *in, T *out, uint32_t n, T *left_cache, T *right_cache,
                 T pivot){

    uint32_t left_i = 0;
    uint32_t right_i = n - BLOCK_SIZE;
    uint32_t status = 0;

    int64_t i = 0;
    int64_t j = n;

    mram_read((__mram_ptr void*)(in), left_cache, BLOCK_SIZE*sizeof(T));
    mram_read((__mram_ptr void*)(in + right_i), right_cache, BLOCK_SIZE*sizeof(T));
    do {
        status = quick_sort_step(left_cache, right_cache, n, &i, &j, pivot);

        if (status == 1 || status == 2) {
            mram_write(left_cache, (__mram_ptr void*) (out + left_i), BLOCK_SIZE*sizeof(T));
            left_i += BLOCK_SIZE;
            mram_read((__mram_ptr void*)(in + left_i), left_cache, BLOCK_SIZE*sizeof(T));          
        }
        if (status == 1 || status == 3) {
            mram_write(right_cache, (__mram_ptr void*) (out + right_i), BLOCK_SIZE*sizeof(T));
            right_i -= BLOCK_SIZE;
            mram_read((__mram_ptr void*)(in + right_i), right_cache, BLOCK_SIZE*sizeof(T));
        }
    } while (status != 0);

    uint32_t nr_left = i % BLOCK_SIZE;
    uint32_t nr_right = (n - i)%BLOCK_SIZE;
    if (nr_left > 0) {
        mram_write(left_cache, (__mram_ptr void*) (out + left_i), nr_left*sizeof(T));
    }
    if (nr_right > 0) {
        mram_write(right_cache + BLOCK_SIZE - nr_right, (__mram_ptr void*) (out + right_i + BLOCK_SIZE - nr_right), nr_right*sizeof(T));
    }

    return i;
}

void sort_full(T *in, T *out, uint32_t n, T *local_cache, uint32_t rand) {
    if (n <= 2*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) in, local_cache, n*sizeof(T));
        quick_sort(local_cache, n);
        mram_write(local_cache, (__mram_ptr void*) out, n*sizeof(uint64_t));
    }
    else {
        T pivots[5];
        mram_read((__mram_ptr void*) (in+rand), pivots, 5*sizeof(T));
        selection_sort(pivots, 5);
        uint64_t index = sort_blocks(in, out, n, local_cache, local_cache+BLOCK_SIZE, pivots[3]);

        if (index > 1) {
            sort_full(out, out, index, local_cache, (5*rand + 1)%BLOCK_SIZE);
        }
        if (n - index > 1) {
            sort_full(out + index, out + index, n - index, local_cache, (5*rand + 1)%BLOCK_SIZE);
        }
    }
}

void reorder(T *input, T *output, T* local_cache, uint32_t tasklet_id,
             uint64_t indices_loc[NR_SPLITS][NR_TASKLETS], uint64_t indices_off[NR_SPLITS][NR_TASKLETS]) {

    // The offset to load the elements
    uint64_t offset_in = 0;
    // The offset to store the elements
    uint64_t offset_glob = 0;
    for (uint32_t i = 0; i < NR_SPLITS; i++) {
        uint64_t offset = 0;
        uint64_t length = 0;
        if (tasklet_id > 0) {
            mram_read((__mram_ptr void*) &indices_off[i][tasklet_id-1], &offset, sizeof(uint64_t));
        }
        mram_read((__mram_ptr void*) &indices_loc[i][tasklet_id], &length, sizeof(uint64_t));

        uint64_t offset_tot = offset + offset_glob;

        for (uint32_t base = 0; base + 2*BLOCK_SIZE <= length; base += 2*BLOCK_SIZE) {
            mram_read((__mram_ptr void*) (input + offset_in), local_cache, 2*BLOCK_SIZE*sizeof(T));

            mram_write(local_cache, (__mram_ptr void*) (output + offset_tot), 2*BLOCK_SIZE*sizeof(T));
            offset_in += 2*BLOCK_SIZE;
            offset_tot += 2*BLOCK_SIZE;
        }
        uint64_t rem_size = length % (2*BLOCK_SIZE);
        if (rem_size > 0) {
            mram_read((__mram_ptr void*) (input + offset_in), local_cache, rem_size*sizeof(T));

            mram_write(local_cache, (__mram_ptr void*) (output + offset_tot), rem_size*sizeof(T));
        }

        offset_in += rem_size;
        
        uint64_t offset_glob_tmp;
        mram_read((__mram_ptr void*) &indices_off[i][NR_TASKLETS-1], &offset_glob_tmp, sizeof(uint64_t));
        offset_glob += offset_glob_tmp;
    }
}