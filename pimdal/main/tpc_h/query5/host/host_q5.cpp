#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cstring>

#include <arrow/api.h>
#include <arrow/acero/exec_plan.h>
#include <arrow/dataset/api.h>
#include "../../reader/read_table.cpp"

#include "param.h"
#include "datatype.h"
#include "transfer_helper.h"

#ifndef NR_DPU
#define NR_DPU 4
#endif

namespace ac = arrow::acero;
namespace cp = arrow::compute;

std::shared_ptr<arrow::Table> customer;
std::shared_ptr<arrow::Table> orders;
std::shared_ptr<arrow::Table> lineitem;
std::shared_ptr<arrow::Table> supplier;
std::shared_ptr<arrow::Table> nation;
std::shared_ptr<arrow::Table> region;

void populate_mram_1(dpu_set_t &system) {
    copy_table(system, region, "r_regionkey", "r_regionkey", 0, DPU_XFER_DEFAULT);
    copy_table(system, region, "r_name", "r_name", 0, DPU_XFER_DEFAULT);
    copy_table(system, nation, "n_nationkey", "n_nationkey", 0, DPU_XFER_DEFAULT);
    copy_table(system, nation, "n_regionkey", "n_regionkey", 0, DPU_XFER_DEFAULT);
    copy_table(system, nation, "n_name", "n_name", 0, DPU_XFER_DEFAULT);

    scatter_table(system, customer, "c_nationkey", "c_nationkey", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, customer, "c_custkey", "c_custkey", 0, DPU_SG_XFER_DEFAULT);

    std::vector<std::vector<query_args_t>> query_args {NR_DPU, std::vector<query_args_t>(1)};
    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        if (customer->num_rows() % NR_DPU == 0) {
            query_args[dpu][0].c_count = customer->num_rows()/NR_DPU;
        }
        else {
            if (dpu < NR_DPU - 1) {
                query_args[dpu][0].c_count = customer->num_rows()/NR_DPU + 1;
            }
            else {
                query_args[dpu][0].c_count = customer->num_rows() % (customer->num_rows()/NR_DPU + 1);
            }
        }

        query_args[dpu][0].r_count = region->num_rows();
        query_args[dpu][0].n_count = nation->num_rows();

        strcpy(query_args[dpu][0].r_region, "ASIA");
    }

    dist_vec(system, query_args, 0, "dpu_args", DPU_XFER_DEFAULT);
}

void populate_mram_2(dpu_set_t &system) {
    scatter_table(system, orders, "o_orderkey", "o_orderkey", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, orders, "o_custkey", "o_custkey", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, orders, "o_orderdate", "o_orderdate", 0, DPU_SG_XFER_DEFAULT);

    std::vector<std::vector<query_args_t>> query_args {NR_DPU, std::vector<query_args_t>(1)};
    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        if (orders->num_rows() % NR_DPU == 0) {
            query_args[dpu][0].o_count = orders->num_rows()/NR_DPU;
        }
        else {
            if (dpu < NR_DPU - 1) {
                query_args[dpu][0].o_count = orders->num_rows()/NR_DPU + 1;
            }
            else {
                query_args[dpu][0].o_count = orders->num_rows() % (orders->num_rows()/NR_DPU + 1);
            }
        }

        query_args[dpu][0].date_start = date_to_int("1994-01-01");
        query_args[dpu][0].date_end = date_to_int("1995-01-01");
    }

    dist_vec(system, query_args, 0, "dpu_args", DPU_XFER_DEFAULT);
}

void populate_mram_3(dpu_set_t &system) {
    scatter_table(system, lineitem, "l_orderkey", "l_orderkey", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, lineitem, "l_suppkey", "l_suppkey", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, lineitem, "l_extendedprice", "l_extendedprice", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, lineitem, "l_discount", "l_discount", 0, DPU_SG_XFER_DEFAULT);

    std::vector<std::vector<query_args_t>> query_args {NR_DPU, std::vector<query_args_t>(1)};
    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        if (lineitem->num_rows() % NR_DPU == 0) {
            query_args[dpu][0].l_count = lineitem->num_rows()/NR_DPU;
        }
        else {
            if (dpu < NR_DPU - 1) {
                query_args[dpu][0].l_count = lineitem->num_rows()/NR_DPU + 1;
            }
            else {
                query_args[dpu][0].l_count = lineitem->num_rows() % (lineitem->num_rows()/NR_DPU + 1);
            }
        }
    }

    dist_vec(system, query_args, 0, "dpu_args", DPU_XFER_DEFAULT);
}

void populate_mram_4(dpu_set_t &system) {
    scatter_table(system, supplier, "s_suppkey", "s_suppkey", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, supplier, "s_nationkey", "s_nationkey", 0, DPU_SG_XFER_DEFAULT);

    std::vector<std::vector<query_args_t>> query_args {NR_DPU, std::vector<query_args_t>(1)};
    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        if (supplier->num_rows() % NR_DPU == 0) {
            query_args[dpu][0].s_count = supplier->num_rows()/NR_DPU;
        }
        else {
            if (dpu < NR_DPU - 1) {
                query_args[dpu][0].s_count = supplier->num_rows()/NR_DPU + 1;
            }
            else {
                query_args[dpu][0].s_count = supplier->num_rows() % (supplier->num_rows()/NR_DPU + 1);
            }
        }
    }

    dist_vec(system, query_args, 0, "dpu_args", DPU_XFER_DEFAULT);
}

void distribute_1(dpu_set_t &system, std::vector<std::vector<uint64_t>> sizes_c,
                  std::vector<std::vector<uint64_t>> sizes_o,
                  arrow::BufferVector buf_c_custkey,
                  arrow::BufferVector buf_o_custkey,
                  arrow::BufferVector buf_c_nationkey,
                  arrow::BufferVector buf_o_orderkey) {
    
    std::vector<std::vector<query_args_t>> dpu_args (NR_DPU, std::vector<query_args_t>(1));
    std::vector<std::vector<uint64_t>> off_data (NR_DPU, std::vector<uint64_t>(NR_DPU+1));
    std::vector<std::vector<uint64_t>> off_data_c (NR_DPU, std::vector<uint64_t>(NR_DPU+1));

    uint32_t max_inner_size = 0;
    uint32_t max_outer_size = 0;
    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        for (uint32_t src = 0; src < NR_DPU; src++) {
            dpu_args[dpu][0].c_count += sizes_c[src][dpu+1] - sizes_c[src][dpu];
            dpu_args[dpu][0].o_count += sizes_o[src][dpu+1] - sizes_o[src][dpu];
        }

        if (dpu_args[dpu][0].c_count > max_inner_size) {
            max_inner_size = dpu_args[dpu][0].c_count;
        }

        if (dpu_args[dpu][0].o_count > max_outer_size) {
            max_outer_size = dpu_args[dpu][0].o_count;
        }
    }

    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        for (uint32_t i = 0; i < NR_DPU+1; i++) {
            off_data[dpu][i] = sizes_o[dpu][i] * sizeof(uint32_t);
            off_data_c[dpu][i] = sizes_c[dpu][i] * sizeof(uint32_t);
            sizes_c[dpu][i] *= sizeof(key_ptr32);
            sizes_o[dpu][i] *= sizeof(key_ptr32);
        }
    }

    // Distributing c_custkey
    sg_xfer_context_2d sc_args_c = {.partitions = buf_c_custkey, .offset = sizes_c};
    get_block_t get_block_info_c = {.f = &get_cpy_ptr_2d, .args = &sc_args_c, .args_size=(sizeof(sc_args_c))};

    dpu_sg_xfer_flags_t flag = dpu_sg_xfer_flags_t(DPU_SG_XFER_DEFAULT | DPU_SG_XFER_DISABLE_LENGTH_CHECK);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
                                max_inner_size*sizeof(key_ptr32), &get_block_info_c, flag));

    // Distributing o_custkey
    sg_xfer_context_2d sc_args_o = {.partitions = buf_o_custkey, .offset = sizes_o};
    get_block_t get_block_info_o = {.f = &get_cpy_ptr_2d, .args = &sc_args_o, .args_size=(sizeof(sc_args_o))};

    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 524288*sizeof(key_ptr32),
                                max_outer_size*sizeof(key_ptr32), &get_block_info_o, flag));

    // Distributing c_nationkey
    sg_xfer_context_2d sc_args_nationkey = {.partitions = buf_c_nationkey, .offset = off_data_c};
    get_block_t get_block_info_nationkey = {.f = &get_cpy_ptr_2d, .args = &sc_args_nationkey, .args_size=(sizeof(sc_args_nationkey))};

    uint32_t length = (max_inner_size*sizeof(uint32_t) + 7) & (-8);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 2*524288*sizeof(key_ptr32),
                                length, &get_block_info_nationkey, flag));

    // Distributing o_orderkey
    sg_xfer_context_2d sc_args_orderkey = {.partitions = buf_o_orderkey, .offset = off_data};
    get_block_t get_block_info_orderkey = {.f = &get_cpy_ptr_2d, .args = &sc_args_orderkey, .args_size=(sizeof(sc_args_orderkey))};

    length = (max_outer_size*sizeof(uint32_t) + 7) & (-8);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 3*524288*sizeof(key_ptr32),
                                length, &get_block_info_orderkey, flag));

    dist_vec(system, dpu_args, 0, "dpu_args", DPU_XFER_DEFAULT);

}

void distribute_2(dpu_set_t &system, std::vector<std::vector<uint64_t>> sizes_o,
                  std::vector<std::vector<uint64_t>> sizes_l,
                  arrow::BufferVector buf_o_orderkey,
                  arrow::BufferVector buf_l_orderkey,
                  arrow::BufferVector buf_o_nationkey,
                  arrow::BufferVector buf_l_suppkey,
                  arrow::BufferVector buf_l_extendedprice,
                  arrow::BufferVector buf_l_discount) {
    
    std::vector<std::vector<query_args_t>> dpu_args (NR_DPU, std::vector<query_args_t>(1));
    std::vector<std::vector<uint64_t>> off_key_o (NR_DPU, std::vector<uint64_t>(NR_DPU+1));
    std::vector<std::vector<uint64_t>> off_key (NR_DPU, std::vector<uint64_t>(NR_DPU+1));
    std::vector<std::vector<uint64_t>> off_data (NR_DPU, std::vector<uint64_t>(NR_DPU+1));

    uint32_t max_inner_size = 0;
    uint32_t max_outer_size = 0;
    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        for (uint32_t src = 0; src < NR_DPU; src++) {
            dpu_args[dpu][0].o_count += sizes_o[src][dpu+1] - sizes_o[src][dpu];
            dpu_args[dpu][0].l_count += sizes_l[src][dpu+1] - sizes_l[src][dpu];
        }

        if (dpu_args[dpu][0].o_count > max_inner_size) {
            max_inner_size = dpu_args[dpu][0].o_count;
        }

        if (dpu_args[dpu][0].l_count > max_outer_size) {
            max_outer_size = dpu_args[dpu][0].l_count;
        }
    }

    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        for (uint32_t i = 0; i < NR_DPU+1; i++) {
            off_key_o[dpu][i] = sizes_o[dpu][i] * sizeof(uint32_t);
            off_data[dpu][i] = sizes_l[dpu][i] * sizeof(int64_t);
            off_key[dpu][i] = sizes_l[dpu][i] * sizeof(uint32_t);
            sizes_o[dpu][i] *= sizeof(key_ptr32);
            sizes_l[dpu][i] *= sizeof(key_ptr32);
        }
    }

    // Distributing o_orderkey
    sg_xfer_context_2d sc_args_o = {.partitions = buf_o_orderkey, .offset = sizes_o};
    get_block_t get_block_info_o = {.f = &get_cpy_ptr_2d, .args = &sc_args_o, .args_size=(sizeof(sc_args_o))};

    dpu_sg_xfer_flags_t flag = dpu_sg_xfer_flags_t(DPU_SG_XFER_DEFAULT | DPU_SG_XFER_DISABLE_LENGTH_CHECK);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
                                max_inner_size*sizeof(key_ptr32), &get_block_info_o, flag));

    // Distributing l_orderkey
    sg_xfer_context_2d sc_args_l = {.partitions = buf_l_orderkey, .offset = sizes_l};
    get_block_t get_block_info_l = {.f = &get_cpy_ptr_2d, .args = &sc_args_l, .args_size=(sizeof(sc_args_l))};

    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 524288*sizeof(key_ptr32),
                                max_outer_size*sizeof(key_ptr32), &get_block_info_l, flag));

    // Distributing o_nationkey
    sg_xfer_context_2d sc_args_nationkey = {.partitions = buf_o_nationkey, .offset = off_key_o};
    get_block_t get_block_info_nationkey = {.f = &get_cpy_ptr_2d, .args = &sc_args_nationkey, .args_size=(sizeof(sc_args_nationkey))};

    uint32_t length = (max_inner_size + 1) & (-2);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 2*524288*sizeof(key_ptr32),
                                length*sizeof(uint32_t), &get_block_info_nationkey, flag));

    // Distributing l_suppkey
    sg_xfer_context_2d sc_args_suppkey = {.partitions = buf_l_suppkey, .offset = off_key};
    get_block_t get_block_info_suppkey = {.f = &get_cpy_ptr_2d, .args = &sc_args_suppkey, .args_size=(sizeof(sc_args_suppkey))};

    length = (max_outer_size + 1) & (-2);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 3*524288*sizeof(key_ptr32),
                                length*sizeof(uint32_t), &get_block_info_suppkey, flag));

    // Distributing l_extendedprice
    sg_xfer_context_2d sc_args_extendedprice = {.partitions = buf_l_extendedprice, .offset = off_data};
    get_block_t get_block_info_extendedprice = {.f = &get_cpy_ptr_2d, .args = &sc_args_extendedprice, .args_size=(sizeof(sc_args_extendedprice))};

    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 4*524288*sizeof(key_ptr32),
                                max_outer_size*sizeof(int64_t), &get_block_info_extendedprice, flag));

    // Distributing l_discount
    sg_xfer_context_2d sc_args_discount = {.partitions = buf_l_discount, .offset = off_data};
    get_block_t get_block_info_discount = {.f = &get_cpy_ptr_2d, .args = &sc_args_discount, .args_size=(sizeof(sc_args_discount))};

    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 5*524288*sizeof(key_ptr32),
                                max_outer_size*sizeof(int64_t), &get_block_info_discount, flag));

    dist_vec(system, dpu_args, 0, "dpu_args", DPU_XFER_DEFAULT);

}

void distribute_3(dpu_set_t &system, std::vector<std::vector<uint64_t>> sizes_s,
                  std::vector<std::vector<uint64_t>> sizes_l,
                  arrow::BufferVector buf_s_suppkey,
                  arrow::BufferVector buf_l_suppkey,
                  arrow::BufferVector buf_s_nationkey,
                  arrow::BufferVector buf_l_nationkey,
                  arrow::BufferVector buf_l_extendedprice,
                  arrow::BufferVector buf_l_discount) {
    
    std::vector<std::vector<query_args_t>> dpu_args (NR_DPU, std::vector<query_args_t>(1));
    std::vector<std::vector<uint64_t>> off_key_s (NR_DPU, std::vector<uint64_t>(NR_DPU+1));
    std::vector<std::vector<uint64_t>> off_data (NR_DPU, std::vector<uint64_t>(NR_DPU+1));
    std::vector<std::vector<uint64_t>> off_key (NR_DPU, std::vector<uint64_t>(NR_DPU+1));

    uint32_t max_inner_size = 0;
    uint32_t max_outer_size = 0;
    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        for (uint32_t src = 0; src < NR_DPU; src++) {
            dpu_args[dpu][0].s_count += sizes_s[src][dpu+1] - sizes_s[src][dpu];
            dpu_args[dpu][0].l_count += sizes_l[src][dpu+1] - sizes_l[src][dpu];
        }

        if (dpu_args[dpu][0].s_count > max_inner_size) {
            max_inner_size = dpu_args[dpu][0].s_count;
        }

        if (dpu_args[dpu][0].l_count > max_outer_size) {
            max_outer_size = dpu_args[dpu][0].l_count;
        }
    }

    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        for (uint32_t i = 0; i < NR_DPU+1; i++) {
            off_key_s[dpu][i] = sizes_s[dpu][i] * sizeof(uint32_t);
            off_data[dpu][i] = sizes_l[dpu][i] * sizeof(int64_t);
            off_key[dpu][i] = sizes_l[dpu][i] * sizeof(uint32_t);
            sizes_s[dpu][i] *= sizeof(key_ptr32);
            sizes_l[dpu][i] *= sizeof(key_ptr32);
        }
    }

    // Distributing s_suppkey
    sg_xfer_context_2d sc_args_s = {.partitions = buf_s_suppkey, .offset = sizes_s};
    get_block_t get_block_info_s = {.f = &get_cpy_ptr_2d, .args = &sc_args_s, .args_size=(sizeof(sc_args_s))};

    dpu_sg_xfer_flags_t flag = dpu_sg_xfer_flags_t(DPU_SG_XFER_DEFAULT | DPU_SG_XFER_DISABLE_LENGTH_CHECK);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
                                max_inner_size*sizeof(key_ptr32), &get_block_info_s, flag));

    // Distributing l_suppkey
    sg_xfer_context_2d sc_args_l = {.partitions = buf_l_suppkey, .offset = sizes_l};
    get_block_t get_block_info_l = {.f = &get_cpy_ptr_2d, .args = &sc_args_l, .args_size=(sizeof(sc_args_l))};

    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 524288*sizeof(key_ptr32),
                                max_outer_size*sizeof(key_ptr32), &get_block_info_l, flag));

    // Distributing s_nationkey
    sg_xfer_context_2d sc_args_s_nationkey = {.partitions = buf_s_nationkey, .offset = off_key_s};
    get_block_t get_block_info_s_nationkey = {.f = &get_cpy_ptr_2d, .args = &sc_args_s_nationkey, .args_size=(sizeof(sc_args_s_nationkey))};

    uint32_t length = (max_inner_size + 1) & (-2);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 2*524288*sizeof(key_ptr32),
                                length*sizeof(uint32_t), &get_block_info_s_nationkey, flag));
    
    // Distributing l_nationkey
    sg_xfer_context_2d sc_args_l_nationkey = {.partitions = buf_l_nationkey, .offset = off_key};
    get_block_t get_block_info_l_nationkey = {.f = &get_cpy_ptr_2d, .args = &sc_args_l_nationkey, .args_size=(sizeof(sc_args_l_nationkey))};

    length = (max_outer_size + 1) & (-2);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 3*524288*sizeof(key_ptr32),
                                length*sizeof(uint32_t), &get_block_info_l_nationkey, flag));

    // Distributing l_extendedprice
    sg_xfer_context_2d sc_args_extendedprice = {.partitions = buf_l_extendedprice, .offset = off_data};
    get_block_t get_block_info_extendedprice = {.f = &get_cpy_ptr_2d, .args = &sc_args_extendedprice, .args_size=(sizeof(sc_args_extendedprice))};

    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 4*524288*sizeof(key_ptr32),
                                max_outer_size*sizeof(int64_t), &get_block_info_extendedprice, flag));

    // Distributing l_discount
    sg_xfer_context_2d sc_args_discount = {.partitions = buf_l_discount, .offset = off_data};
    get_block_t get_block_info_discount = {.f = &get_cpy_ptr_2d, .args = &sc_args_discount, .args_size=(sizeof(sc_args_discount))};

    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 5*524288*sizeof(key_ptr32),
                                max_outer_size*sizeof(int64_t), &get_block_info_discount, flag));

    dist_vec(system, dpu_args, 0, "dpu_args", DPU_XFER_DEFAULT);

}

arrow::BufferVector collect(dpu_set_t &system, uint32_t offset, uint32_t type_size) {
    std::vector<std::vector<query_res_t>> query_res {NR_DPU, std::vector<query_res_t>(1)};

    get_vec(system, query_res, 0, "dpu_results", DPU_XFER_DEFAULT);

    struct dpu_set_t dpu;
    uint32_t each_dpu;
    uint32_t max_sel_size = 0;
    uint64_t tot_count = 0;
    DPU_FOREACH(system, dpu, each_dpu) {
        uint32_t sel_size = query_res[each_dpu][0].count;
        if (sel_size > max_sel_size) {
            max_sel_size = sel_size;
        }

        tot_count += sel_size;
    }
    std::cout << "Max count: " << max_sel_size;
    std::cout << " Total count: " << tot_count << std::endl; 

    uint32_t size_rounded = (max_sel_size*type_size + 7) & (-8);
    arrow::BufferVector buffers = alloc_buf_vec(size_rounded, NR_DPU);
    get_buf(system, buffers, offset, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    return buffers;
}

std::shared_ptr<arrow::Table> get_results(dpu_set_t &system) {
    std::vector<std::vector<query_res_t>> query_res {NR_DPU, std::vector<query_res_t>(1)};
    arrow::ArrayVector key_chunks;
    arrow::ArrayVector rev_chunks;

    get_vec(system, query_res, 0, "dpu_results", DPU_XFER_DEFAULT);

    struct dpu_set_t dpu;
    uint32_t each_dpu;
    uint32_t max_size = 0;
    uint32_t tot = 0;
    DPU_FOREACH(system, dpu, each_dpu) {
        tot += query_res[each_dpu][0].count;
        //std::cout << "Size: " << query_res[each_dpu][0].count << std::endl;
        if (query_res[each_dpu][0].count > max_size) {
            max_size = query_res[each_dpu][0].count;
        }
    }

    uint32_t length = (max_size*sizeof(uint32_t) + 7) & (-8);
    arrow::BufferVector buffers_key = alloc_buf_vec(length, NR_DPU);
    get_buf(system, buffers_key, 0, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    arrow::BufferVector buffers_rev = alloc_buf_vec(max_size*sizeof(int64_t), NR_DPU);
    get_buf(system, buffers_rev, 524288*sizeof(key_ptr32), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    DPU_FOREACH(system, dpu, each_dpu) {
        uint32_t dpu_size = query_res[each_dpu][0].count;
        auto array_data_key = arrow::ArrayData::Make(arrow::int32(), dpu_size, {nullptr, buffers_key[each_dpu]});
        auto array_key = arrow::MakeArray(array_data_key);
        key_chunks.push_back(array_key);

        auto array_data_val = arrow::ArrayData::Make(arrow::int64(), dpu_size, {nullptr, buffers_rev[each_dpu]});
        auto array_val = arrow::MakeArray(array_data_val);
        rev_chunks.push_back(array_val);
    }

    arrow::ChunkedArrayVector data_vec;
    auto schema = arrow::schema({arrow::field("n_nationkey", arrow::int32(), false),
                                 arrow::field("revenue", arrow::int64(), false)});
    data_vec.push_back(std::make_shared<arrow::ChunkedArray>(key_chunks));
    data_vec.push_back(std::make_shared<arrow::ChunkedArray>(rev_chunks));

    return arrow::Table::Make(schema, data_vec);
}

std::shared_ptr<arrow::Table> aggr_host(std::shared_ptr<arrow::Table> res) {    

    auto aggregate_options =
    ac::AggregateNodeOptions{{{"hash_sum", nullptr, "revenue", "revenue"}},
                            {"n_nationkey"}};

    ac::Declaration right = ac::Declaration::Sequence({{"table_source", ac::TableSourceNodeOptions(res)},
                                                      {"aggregate", aggregate_options}});

    ac::Declaration left{"table_source", ac::TableSourceNodeOptions(nation)};
    
    ac::HashJoinNodeOptions join_opts{
        ac::JoinType::INNER,
        {"n_nationkey"},
        {"n_nationkey"}, cp::literal(true), "l_", "r_"
    };
    ac::Declaration joined{
        "hashjoin", {std::move(left), std::move(right)}, std::move(join_opts)
    };

    
    cp::Expression n_nationkey = cp::field_ref("n_name");
    cp::Expression revenue = cp::field_ref("revenue");
    std::vector<arrow::compute::Expression> projection_list = {
        n_nationkey, revenue
    };
    std::vector<std::string> projection_names = {
        "n_name", "revenue"
    };
    ac::ProjectNodeOptions project_opts(std::move(projection_list), std::move(projection_names));
    ac::Declaration proj{"project", {std::move(joined)}, std::move(project_opts)};

    auto order_by_options = ac::OrderByNodeOptions({{arrow::compute::SortKey("revenue", cp::SortOrder::Descending)}});
    ac::Declaration ordered{"order_by", {std::move(proj)}, std::move(order_by_options)};

    ac::QueryOptions query_options;
    query_options.use_threads = true;

    auto query = ac::DeclarationToTable(ordered, query_options);
    auto table = query.ValueOrDie();
    std::cout << table->ToString() << std::endl;

    return table;
}

double comp() {
    auto pred_l = cp::equal(cp::field_ref("r_name"),
                            cp::literal("ASIA"));

    auto pred_o = cp::and_(cp::greater_equal(cp::field_ref("o_orderdate"),
                                             cp::literal(date_to_int("1993-07-01"))),
                           cp::less(cp::field_ref("o_orderdate"),
                                    cp::literal(date_to_int("1993-10-01"))));

    ac::QueryOptions query_options;
    query_options.use_threads = true;

    //auto query = ac::DeclarationToTable(plan, query_options);
    //auto table = query.ValueOrDie();
    //std::cout << table->ToString() << std::endl;

    return 0;
}

int main(void) {
    dpu_set_t system;
    DPU_ASSERT(dpu_alloc(NR_DPU, "sgXferEnable=true", &system));

    std::vector<int32_t> n_cols = {0, 1, 2};
    auto status = parquet_to_table("nation", nation, n_cols);
    if (!status.ok()) {
        std::cout << status.message() << std::endl;
    }

    std::vector<int32_t> r_cols = {0, 1};
    status = parquet_to_table("region", region, r_cols);
    if (!status.ok()) {
        std::cout << status.message() << std::endl;
    }

    std::vector<int32_t> c_cols = {0, 3};
    status = parquet_to_table("customer", customer, c_cols);
    if (!status.ok()) {
        std::cout << status.message() << std::endl;
    }

    std::vector<int32_t> o_cols = {0, 1, 4};
    status = parquet_to_table("orders", orders, o_cols);
    if (!status.ok()) {
        std::cout << status.message() << std::endl;
    }

    std::vector<int32_t> l_cols = {0, 2, 5, 6};
    status = parquet_to_table("lineitem", lineitem, l_cols);
    if (!status.ok()) {
        std::cout << status.message() << std::endl;
    }

    std::vector<int32_t> s_cols = {0, 3};
    status = parquet_to_table("supplier", supplier, s_cols);
    if (!status.ok()) {
        std::cout << status.message() << std::endl;
    }

    try {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        {
            DPU_ASSERT(dpu_load(system, "kernel_q5_1", NULL));
            populate_mram_1(system);
            DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));

            {
                std::vector<std::vector<uint64_t>> sizes_c(NR_DPU, std::vector<uint64_t>(NR_DPU+1));
                get_vec(system, sizes_c, 3*524288*sizeof(key_ptrtext), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);
                auto buf_c_custkey = collect(system, 0, sizeof(key_ptr32));
                auto buf_c_nationkey = collect(system, 524288*sizeof(key_ptrtext), sizeof(uint32_t));

                DPU_ASSERT(dpu_load(system, "kernel_q5_2", NULL));
                populate_mram_2(system);
                DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));
                std::vector<std::vector<uint64_t>> sizes_o(NR_DPU, std::vector<uint64_t>(NR_DPU+1));
                get_vec(system, sizes_o, 4*524288*sizeof(key_ptr32), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);
                auto buf_o_custkey = collect(system, 0, sizeof(key_ptr32));
                auto buf_o_orderkey = collect(system, 524288*sizeof(key_ptr32), sizeof(uint32_t));

                DPU_ASSERT(dpu_load(system, "kernel_q5_3", NULL));
                distribute_1(system, sizes_c, sizes_o, buf_c_custkey, buf_o_custkey,
                            buf_c_nationkey, buf_o_orderkey);
            }

            DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));

            {
                std::vector<std::vector<uint64_t>> sizes_o(NR_DPU, std::vector<uint64_t>(NR_DPU+1));
                get_vec(system, sizes_o, 6*524288*sizeof(key_ptr32), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);
                auto buf_o_orderkey = collect(system, 0, sizeof(key_ptr32));
                auto buf_o_nationkey = collect(system, 524288*sizeof(key_ptr32), sizeof(uint32_t));

                DPU_ASSERT(dpu_load(system, "kernel_q5_4", NULL));
                populate_mram_3(system);
                DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));
                std::vector<std::vector<uint64_t>> sizes_l(NR_DPU, std::vector<uint64_t>(NR_DPU+1));
                get_vec(system, sizes_l, 4*524288*sizeof(key_ptr32), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);
                auto buf_l_orderkey = collect(system, 0, sizeof(key_ptr32));
                auto buf_l_suppkey = collect(system, 524288*sizeof(key_ptr32), sizeof(uint32_t));
                auto buf_l_extendedprice = collect(system, 2*524288*sizeof(key_ptr32), sizeof(int64_t));
                auto buf_l_discount = collect(system, 3*524288*sizeof(key_ptr32), sizeof(int64_t));

                DPU_ASSERT(dpu_load(system, "kernel_q5_5", NULL));
                distribute_2(system, sizes_o, sizes_l, buf_o_orderkey, buf_l_orderkey,
                            buf_o_nationkey, buf_l_suppkey, buf_l_extendedprice,
                            buf_l_discount);
            }

            DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));

            {
                std::vector<std::vector<uint64_t>> sizes_l(NR_DPU, std::vector<uint64_t>(NR_DPU+1));
                get_vec(system, sizes_l, 8*524288*sizeof(key_ptr32), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);
                auto buf_l_suppkey = collect(system, 0, sizeof(key_ptr32));
                auto buf_l_nationkey = collect(system, 524288*sizeof(key_ptr32), sizeof(uint32_t));
                auto buf_l_discount = collect(system, 2*524288*sizeof(key_ptr32), sizeof(int64_t));
                auto buf_l_extendedprice = collect(system, 3*524288*sizeof(key_ptr32), sizeof(int64_t));

                DPU_ASSERT(dpu_load(system, "kernel_q5_6", NULL));
                populate_mram_4(system);
                DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));
                std::vector<std::vector<uint64_t>> sizes_s(NR_DPU, std::vector<uint64_t>(NR_DPU+1));
                get_vec(system, sizes_s, 2*524288*sizeof(key_ptr32), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);
                auto buf_s_suppkey = collect(system, 0, sizeof(key_ptr32));
                auto buf_s_nationkey = collect(system, 524288*sizeof(key_ptr32), sizeof(uint32_t));

                DPU_ASSERT(dpu_load(system, "kernel_q5_7", NULL));
                distribute_3(system, sizes_s, sizes_l, buf_s_suppkey, buf_l_suppkey,
                                buf_s_nationkey, buf_l_nationkey,
                                buf_l_extendedprice, buf_l_discount);
            }

            DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));

            auto res = get_results(system);
            aggr_host(res);
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "Host elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " millisecs." << std::endl;

        output_dpu(system);
    }
    catch (const dpu::DpuError & e) {
        std::cerr << e.what() << std::endl;
    }
    
    return 0;
}