#include <defs.h>
#include <barrier.h>
#include <mram.h>
#include <alloc.h>
#include <stdint.h>
#include <stdio.h>
#include <mutex.h>
#include <mutex_pool.h>

#include "datatype.h"
#include "param.h"
#include "sel.h"
#include "arithmetic.h"
#include "reduce.h"

#define BLOCK_SIZE 64
#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif

__host query_args_t dpu_args;
__host query_res_t dpu_results;

sel_results_t sel_results;
red_results_t red_results;

__mram_noinit_keep int64_t l_extendedprice[524288];
__mram_noinit_keep uint32_t l_shipdate[524288];
__mram_noinit_keep int64_t l_discount[524288];
__mram_noinit_keep int64_t l_quantity[524288];

BARRIER_INIT(barrier, NR_TASKLETS);
MUTEX_INIT(mutex);

void out_val(key_ptr_t *in, int64_t *out) {
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
        out[i] = in[i].key;
    }
}

void create_ptr(uint32_t in, uint32_t out, uint32_t count) {

    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    uint32_t* key_cache = (uint32_t*) mem_alloc(BLOCK_SIZE*sizeof(uint32_t));
    key_ptr_t* ptr_cache = (key_ptr_t*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        mram_read((__mram_ptr void*) (in + base*sizeof(uint32_t)), key_cache, BLOCK_SIZE*sizeof(uint32_t));

        for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
            key_ptr_t new_ptr = {.key = key_cache[i], .ptr = base+i};
            ptr_cache[i] = new_ptr;
        }

        mram_write(ptr_cache, (__mram_ptr void*) (out + base*sizeof(key_ptr_t)), BLOCK_SIZE*sizeof(key_ptr_t));
    }
}

void load_next(uint32_t in, uint32_t out, uint32_t load, uint32_t count) {
    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptr_t* sel_cache = (key_ptr_t*) mem_alloc(BLOCK_SIZE*sizeof(key_ptr_t));

    uint32_t base_tasklet = tasklet_id*BLOCK_SIZE;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*BLOCK_SIZE) {
        uint32_t size_load = base + BLOCK_SIZE > count ? count % BLOCK_SIZE : BLOCK_SIZE;

        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr_t)), sel_cache, size_load*sizeof(key_ptr_t));

        for (uint32_t i = 0; i < size_load; i++) {
            // Load the next element into the cache
            mram_read((__mram_ptr void*) (load + sel_cache[i].ptr*sizeof(int64_t)), &sel_cache[i].key, sizeof(int64_t));
        }

        mram_write(sel_cache, (__mram_ptr void*) (out + base*sizeof(key_ptr_t)), size_load*sizeof(key_ptr_t));
    }
}

bool pred_date(key_ptr_t date) {
    return date.key >= dpu_args.date_start && date.key < dpu_args.date_end;
}

bool pred_discount(key_ptr_t discount) {
    return discount.key >= dpu_args.discount - 1 && discount.key <= dpu_args.discount + 1;
}

bool pred_quantity(key_ptr_t quantity) {
    return quantity.key < dpu_args.quantity;
}

int main() {

    uint32_t size = 524288;
    uint32_t buffer_1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t buffer_2 = (uint32_t) (buffer_1 + size*sizeof(key_ptr_t));

    /*
    * l_shipdate >= DATE and l_shipdate < DATE + 1 year
    */
    create_ptr((uint32_t) l_shipdate, buffer_1, dpu_args.size);
    barrier_wait(&barrier);

    sel_arguments_t sel_args = {.in = buffer_1, .out = buffer_2, .pred = &pred_date, .size = dpu_args.size};
    sel_kernel(&sel_args, &sel_results);
    barrier_wait(&barrier);

    /*
    * l_quantity < QUANTITY
    */
    load_next(buffer_2, buffer_1, (uint32_t) l_quantity, sel_results.t_count);
    barrier_wait(&barrier);

    sel_args.pred = &pred_quantity;
    sel_args.size = sel_results.t_count;
    sel_kernel(&sel_args, &sel_results);
    barrier_wait(&barrier);

    /*
    * l_discount between DISCOUNT - 0.01 and DISCOUNT + 0.01
    */
    load_next(buffer_2, buffer_1, (uint32_t) l_discount, sel_results.t_count);
    barrier_wait(&barrier);

    sel_args.pred = &pred_discount;
    sel_args.size = sel_results.t_count;
    sel_kernel(&sel_args, &sel_results);
    barrier_wait(&barrier);

    load_next(buffer_2, buffer_1, (uint32_t) l_extendedprice, sel_results.t_count);
    barrier_wait(&barrier);

    /*
    * sum(l_extendedprice * l_discount)
    */
    ar_arguments_t ar_args = {.buffer_A = buffer_1, .buffer_B = buffer_2,
                              .buffer_out = buffer_2, .size = sel_results.t_count};
    arithmetic_kernel(&ar_args);
    barrier_wait(&barrier);

    red_arguments_t red_args = {.in = buffer_2, .size = sel_results.t_count};
    reduce_kernel(&red_args, &red_results);
    barrier_wait(&barrier);

    dpu_results.revenue = red_results.sum;

    return 0;
}