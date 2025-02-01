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

#define BLOCK_SIZE 64
#ifndef NR_TASKLETS
#define NR_TASKLETS 4
#endif

__host query_args_t dpu_args;
__host query_res_t dpu_results;

sel_results_t sel_results;

__mram_noinit_keep int64_t l_extendedprice[524288];
__mram_noinit_keep int64_t l_discount[524288];
__mram_noinit_keep int64_t l_quantity[524288];
__mram_noinit_keep int64_t l_tax[524288];
__mram_noinit_keep char l_returnflag[524288];
__mram_noinit_keep char l_linestatus[524288];
__mram_noinit_keep uint32_t l_shipdate[524288];

BARRIER_INIT(barrier, NR_TASKLETS);

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

void load_next(uint32_t in, uint32_t out, uint32_t load_flag, uint32_t load_status,
               uint32_t load_qty, uint32_t load_price, uint32_t load_disc,
               uint32_t load_tax, uint32_t count) {
    uint32_t tasklet_id = me();
    if (tasklet_id == 0){
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&barrier);

    key_ptr32* sel_cache = (key_ptr32*) mem_alloc(16*sizeof(key_ptr32));
    char* char_cache = (char*) mem_alloc(24);
    int64_t* dec_cache = (int64_t*) mem_alloc(16*sizeof(int64_t));
    key_ptrout* out_cache = (key_ptrout*) mem_alloc(16*sizeof(key_ptrout));

    uint32_t base_tasklet = tasklet_id*16;
    for (uint32_t base = base_tasklet; base < count; base += NR_TASKLETS*16) {
        uint32_t size_load = base + 16 > count ? count % 16 : 16;

        mram_read((__mram_ptr void*) (in + base*sizeof(key_ptr32)), sel_cache, size_load*sizeof(key_ptr32));

        // Load l_returnflag
        uint32_t i = 0;
        do {
            uint32_t start = sel_cache[i].ptr;
            uint32_t offset = start & 7;
            mram_read((__mram_ptr void*) (load_flag + start - offset), char_cache, 24);
            for (uint32_t j = 0; j < size_load; j++) {
                if (sel_cache[i].ptr == start + j){
                    out_cache[i].l_returnflag = char_cache[j + offset];
                    i++;
                }
            }
        } while (i < size_load);

        // Load l_linestatus
        i = 0;
        do {
            uint32_t start = sel_cache[i].ptr;
            uint32_t offset = start & 7;
            mram_read((__mram_ptr void*) (load_status + start - offset), char_cache, 24);
            for (uint32_t j = 0; j < size_load; j++) {
                if (sel_cache[i].ptr == start + j){
                    out_cache[i].l_linestatus = char_cache[j + offset];
                    i++;
                }
            }
        } while (i < size_load);

        // Load l_quantity
        i = 0;
        do {
            uint32_t start = sel_cache[i].ptr;
            mram_read((__mram_ptr void*) (load_qty + start*sizeof(int64_t)), dec_cache, 16*sizeof(int64_t));
            for (uint32_t j = 0; j < size_load; j++) {
                if (sel_cache[i].ptr == start + j){
                    out_cache[i].sum_qty = dec_cache[j];
                    i++;
                }
            }
        } while (i < size_load);

        // Load l_extendedprice
        i = 0;
        do {
            uint32_t start = sel_cache[i].ptr;
            mram_read((__mram_ptr void*) (load_price + start*sizeof(int64_t)), dec_cache, 16*sizeof(int64_t));
            for (uint32_t j = 0; j < size_load; j++) {
                if (sel_cache[i].ptr == start + j){
                    out_cache[i].sum_base_price = dec_cache[j];
                    i++;
                }
            }
        } while (i < size_load);

        // Load l_discount
        i = 0;
        do {
            uint32_t start = sel_cache[i].ptr;
            mram_read((__mram_ptr void*) (load_disc + start*sizeof(int64_t)), dec_cache, 16*sizeof(int64_t));
            for (uint32_t j = 0; j < size_load; j++) {
                if (sel_cache[i].ptr == start + j){
                    out_cache[i].avg_disc = dec_cache[j];
                    out_cache[i].sum_disc_price = out_cache[i].sum_base_price * (100 - out_cache[i].avg_disc);
                    i++;
                }
            }
        } while (i < size_load);

        // Load l_tax
        i = 0;
        do {
            uint32_t start = sel_cache[i].ptr;
            mram_read((__mram_ptr void*) (load_tax + start*sizeof(int64_t)), dec_cache, 16*sizeof(int64_t));
            for (uint32_t j = 0; j < size_load; j++) {
                if (sel_cache[i].ptr == start + j){
                    out_cache[i].sum_charge = out_cache[i].sum_disc_price * (100 + dec_cache[j]);
                    i++;
                }
            }
        } while (i < size_load);

        for (i = 0; i < size_load; i++) {
            out_cache[i].count_order = 1;
        }

        mram_write(out_cache, (__mram_ptr void*) (out + base*sizeof(key_ptrout)), size_load*sizeof(key_ptrout));
    }
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

bool pred_date(key_ptr_t date) {
    return date.key < dpu_args.date;
}

int main() {

    uint32_t tasklet_id = me();

    uint32_t size = 524288;
    uint32_t buffer_1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t buffer_2 = (uint32_t) (buffer_1 + size*sizeof(key_ptr32));

    /*
    * l_shipdate < DATE
    */
    create_ptr((uint32_t) l_shipdate, buffer_2, dpu_args.l_count);
    barrier_wait(&barrier);

    sel_arguments_t sel_args = {.in = buffer_2, .out = buffer_1, .pred = &pred_date, .size = dpu_args.l_count};
    sel_kernel(&sel_args, &sel_results);
    barrier_wait(&barrier);

    load_next(buffer_1, buffer_2, (uint32_t) l_returnflag, (uint32_t) l_linestatus,
              (uint32_t) l_quantity, (uint32_t) l_extendedprice, (uint32_t) l_discount,
              (uint32_t) l_tax, sel_results.t_count);
    barrier_wait(&barrier);

    if (tasklet_id == 0) {
        uint64_t count = sel_results.t_count;
        mram_write(&count, (__mram_ptr void*) buffer_1, sizeof(uint64_t));
    }

    return 0;
}