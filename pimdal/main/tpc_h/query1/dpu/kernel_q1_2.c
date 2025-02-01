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
#include "aggregate.h"

#define BLOCK_SIZE 64
#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif

__host query_args_t dpu_args;
__host query_res_t dpu_results;

aggr_results_t aggr_results;

__mram_noinit_keep int64_t l_extendedprice[524288];
__mram_noinit_keep int64_t l_discount[524288];
__mram_noinit_keep int64_t l_quantity[524288];
__mram_noinit_keep int64_t l_tax[524288];
__mram_noinit_keep char l_returnflag[524288];
__mram_noinit_keep char l_linestatus[524288];
__mram_noinit_keep uint32_t l_shipdate[524288];

BARRIER_INIT(barrier, NR_TASKLETS);

MUTEX_INIT(mutex);
MUTEX_POOL_INIT(mutexes, 16);

key_ptrout sum(key_ptrout curr_val, key_ptrout element) {
    curr_val.sum_qty += element.sum_qty;
    curr_val.sum_base_price += element.sum_base_price;
    curr_val.sum_disc_price += element.sum_disc_price;
    curr_val.sum_charge += element.sum_charge;
    curr_val.avg_disc += element.avg_disc;
    curr_val.count_order++;

    return curr_val;
}

void out_val(uint32_t in, uint32_t count) {
    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptrout* out_cache = (key_ptrout*) mem_alloc(16*sizeof(key_ptrout));

    uint32_t base_tasklet = tasklet_id*16;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*16) {
        uint32_t size_load = base + 16 > count ? count % 16 : 16;

        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptrout)), out_cache, size_load);

        for (uint32_t i = 0; i < size_load; i++) {
            printf("%c %c %lld %lld %lld %lld %lld %u\n", out_cache[i].l_returnflag,
                    out_cache[i].l_linestatus, out_cache[i].sum_qty,
                    out_cache[i].sum_base_price, out_cache[i].sum_disc_price,
                    out_cache[i].sum_charge, out_cache[i].avg_disc,
                    out_cache[i].count_order);
        }
    }
}

void load_out(uint32_t in, uint32_t out_flag, uint32_t out_status, uint32_t out_qty,
              uint32_t out_base, uint32_t out_disc, uint32_t out_charge,
              uint32_t out_avg_disc, uint32_t out_count, uint32_t count) {

    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptrout* in_cache = (key_ptrout*) mem_alloc(16*sizeof(key_ptrout));
    char* char_cache = (char*) mem_alloc(24);
    int64_t* dec_cache = (int64_t*) mem_alloc(16*sizeof(int64_t));
    int32_t* int_cache = (int32_t*) mem_alloc(16*sizeof(int32_t));

    uint32_t base_tasklet = tasklet_id*16;
    if (base_tasklet < count) {
        uint32_t size_load = base_tasklet + 16 > count ? count % 16 : 16;

        mram_read((__mram_ptr void*) in, in_cache, size_load*sizeof(key_ptrout));

        for (uint32_t i = 0; i < size_load; i++) {
            char_cache[i] = in_cache[i].l_returnflag;
        }
        uint32_t length = (size_load + 7) & (-8);
        mram_write(char_cache, (__mram_ptr void*) out_flag, length);

        for (uint32_t i = 0; i < size_load; i++) {
            char_cache[i] = in_cache[i].l_linestatus;
        }
        mram_write(char_cache, (__mram_ptr void*) out_status, length);

        for (uint32_t i = 0; i < size_load; i++) {
            dec_cache[i] = in_cache[i].sum_qty;
        }
        mram_write(dec_cache, (__mram_ptr void*) out_qty, size_load*sizeof(int64_t));

        for (uint32_t i = 0; i < size_load; i++) {
            dec_cache[i] = in_cache[i].sum_base_price;
        }
        mram_write(dec_cache, (__mram_ptr void*) out_base, size_load*sizeof(int64_t));

        for (uint32_t i = 0; i < size_load; i++) {
            dec_cache[i] = in_cache[i].sum_disc_price;
        }
        mram_write(dec_cache, (__mram_ptr void*) out_disc, size_load*sizeof(int64_t));

        for (uint32_t i = 0; i < size_load; i++) {
            dec_cache[i] = in_cache[i].sum_charge;
        }
        mram_write(dec_cache, (__mram_ptr void*) out_charge, size_load*sizeof(int64_t));

        for (uint32_t i = 0; i < size_load; i++) {
            dec_cache[i] = in_cache[i].avg_disc;
        }
        mram_write(dec_cache, (__mram_ptr void*) out_avg_disc, size_load*sizeof(int64_t));

        length = (size_load*sizeof(uint32_t) + 7) & (-8);
        for (uint32_t i = 0; i < size_load; i++) {
            int_cache[i] = in_cache[i].count_order;
        }
        mram_write(int_cache, (__mram_ptr void*) out_count, length);
    }
}


int main() {

    uint32_t tasklet_id = me();

    uint32_t size = 524288;
    uint32_t buffer_1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t buffer_2 = (uint32_t) (buffer_1 + size*sizeof(key_ptr32));
    uint32_t out_flag = (uint32_t) (buffer_2);
    uint32_t out_status = (uint32_t) (out_flag + 16*sizeof(int64_t));
    uint32_t out_qty = (uint32_t) (out_status + 16*sizeof(int64_t));
    uint32_t out_base = (uint32_t) (out_qty + 16*sizeof(int64_t));
    uint32_t out_disc = (uint32_t) (out_base + 16*sizeof(int64_t));
    uint32_t out_charge = (uint32_t) (out_disc + 16*sizeof(int64_t));
    uint32_t out_avg_disc = (uint32_t) (out_charge + 16*sizeof(int64_t));
    uint32_t out_count = (uint32_t) (out_avg_disc + 16*sizeof(int64_t));

    if (tasklet_id == 0) {
        uint64_t count;
        mram_read((__mram_ptr void*) buffer_1, &count, sizeof(uint64_t));
        dpu_results.count = count;
    }
    barrier_wait(&barrier);

    aggr_arguments_t aggr_args = {.in = buffer_2, .out = buffer_1, .size = dpu_results.count, .aggr = sum};
    group_kernel(&aggr_args, &aggr_results);

    load_out(buffer_1, out_flag, out_status, out_qty, out_base, out_disc,
             out_charge, out_avg_disc, out_count, aggr_results.t_count);
    
    dpu_results.count = aggr_results.t_count;

    return 0;
}