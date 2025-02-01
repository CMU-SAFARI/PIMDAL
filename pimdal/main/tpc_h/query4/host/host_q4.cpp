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


std::shared_ptr<arrow::Table> lineitem;
std::shared_ptr<arrow::Table> orders;

void populate_mram_1(dpu_set_t &system) {
    scatter_table(system, orders, "o_orderkey", "o_orderkey", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, orders, "o_orderpriority", "o_orderpriority", 0, DPU_SG_XFER_DEFAULT);
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
        
        query_args[dpu][0].date_start = date_to_int("1993-07-01");
        query_args[dpu][0].date_end = date_to_int("1993-10-01");
    }

    dist_vec(system, query_args, 0, "dpu_args", DPU_XFER_DEFAULT);
}

void populate_mram_2(dpu_set_t &system) {
    scatter_table(system, lineitem, "l_orderkey", "l_orderkey", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, lineitem, "l_commitdate", "l_commitdate", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, lineitem, "l_receiptdate", "l_receiptdate", 0, DPU_SG_XFER_DEFAULT);

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

arrow::BufferVector collect(dpu_set_t &system, uint32_t offset, uint32_t type_size) {
    std::vector<std::vector<query_res_t>> query_res {NR_DPU, std::vector<query_res_t>(1)};

    get_vec(system, query_res, 0, "dpu_results", DPU_XFER_DEFAULT);

    struct dpu_set_t dpu;
    uint32_t each_dpu;
    uint32_t max_sel_size = 0;
    uint32_t total = 0;
    DPU_FOREACH(system, dpu, each_dpu) {
        uint32_t sel_size = query_res[each_dpu][0].count;
        if (sel_size > max_sel_size) {
            max_sel_size = sel_size;
        }
        total += sel_size;
    }
    std::cout << "Total count: " << total << std::endl;

    uint32_t size_rounded = (max_sel_size*type_size + 7) & (-8);
    arrow::BufferVector buffers = alloc_buf_vec(size_rounded, NR_DPU);
    get_buf(system, buffers, offset, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    return buffers;
}

void distribute(dpu_set_t &system, std::vector<std::vector<uint64_t>> sizes_o,
                std::vector<std::vector<uint64_t>> sizes_l,
                arrow::BufferVector buf_o_key, arrow::BufferVector buf_l_key,
                arrow::BufferVector buf_o_prio) {
    
    std::vector<std::vector<query_args_t>> dpu_args (NR_DPU, std::vector<query_args_t>(1));
    std::vector<std::vector<uint64_t>> off_oprio (NR_DPU, std::vector<uint64_t>(NR_DPU+1));

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
            off_oprio[dpu][i] = sizes_o[dpu][i] * 16;
            sizes_o[dpu][i] *= sizeof(key_ptr32);
            sizes_l[dpu][i] *= sizeof(key_ptr32);
        }
    }

    // Distributing o_orderkey
    sg_xfer_context_2d sc_args_o = {.partitions = buf_o_key, .offset = sizes_o};
    get_block_t get_block_info_o = {.f = &get_cpy_ptr_2d, .args = &sc_args_o, .args_size=(sizeof(sc_args_o))};

    dpu_sg_xfer_flags_t flag = dpu_sg_xfer_flags_t(DPU_SG_XFER_DEFAULT | DPU_SG_XFER_DISABLE_LENGTH_CHECK);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
                                max_inner_size*sizeof(key_ptr32), &get_block_info_o, flag));

    // Distributing l_orderkey
    sg_xfer_context_2d sc_args_l = {.partitions = buf_l_key, .offset = sizes_l};
    get_block_t get_block_info_l = {.f = &get_cpy_ptr_2d, .args = &sc_args_l, .args_size=(sizeof(sc_args_l))};

    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 524288*sizeof(key_ptr32),
                                max_outer_size*sizeof(key_ptr32), &get_block_info_l, flag));

    // Distributing o_orderpriority
    sg_xfer_context_2d sc_args_oprio = {.partitions = buf_o_prio, .offset = off_oprio};
    get_block_t get_block_info_oprio = {.f = &get_cpy_ptr_2d, .args = &sc_args_oprio, .args_size=(sizeof(sc_args_oprio))};

    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 2*524288*sizeof(key_ptr32),
                                max_inner_size*16, &get_block_info_oprio, flag));

    dist_vec(system, dpu_args, 0, "dpu_args", DPU_XFER_DEFAULT);

}

std::shared_ptr<arrow::Table> get_results(dpu_set_t &system) {
    std::vector<std::vector<query_res_t>> query_res (NR_DPU, std::vector<query_res_t>(1));
    arrow::ArrayVector key_chunks;
    arrow::ArrayVector val_chunks;

    get_vec(system, query_res, 0, "dpu_results", DPU_XFER_DEFAULT);

    struct dpu_set_t dpu;
    uint32_t each_dpu;
    uint32_t max_gb_size = 0;
    uint32_t tot = 0;
    DPU_FOREACH(system, dpu, each_dpu) {
        tot += query_res[each_dpu][0].count;
        uint32_t gb_size = query_res[each_dpu][0].count + (query_res[each_dpu][0].count & 1);
        //std::cout << "Size: " << gb_res[each_dpu][0].count << std::endl;
        if (gb_size > max_gb_size) {
            max_gb_size = gb_size;
        }
    }

    arrow::BufferVector buffers_key = alloc_buf_vec(max_gb_size*16, NR_DPU);
    get_buf(system, buffers_key, 0, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    arrow::BufferVector buffers_val;
    buffers_val = alloc_buf_vec(max_gb_size*sizeof(uint32_t), NR_DPU);
    get_buf(system, buffers_val, 524288*sizeof(key_ptr32), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    DPU_FOREACH(system, dpu, each_dpu) {
        uint32_t dpu_size = query_res[each_dpu][0].count;
        auto array_data_key = arrow::ArrayData::Make(arrow::fixed_size_binary(16), dpu_size, {nullptr, buffers_key[each_dpu]});
        auto array_key = arrow::MakeArray(array_data_key);
        key_chunks.push_back(array_key);

        auto array_data_val = arrow::ArrayData::Make(arrow::uint32(), dpu_size, {nullptr, buffers_val[each_dpu]});
        auto array_val = arrow::MakeArray(array_data_val);
        val_chunks.push_back(array_val);
    }

    arrow::ChunkedArrayVector data_vec;
    auto schema = arrow::schema({arrow::field("o_orderpriority", arrow::fixed_size_binary(16), false),
                                 arrow::field("order_count", arrow::uint32(), false)});
    data_vec.push_back(std::make_shared<arrow::ChunkedArray>(key_chunks));
    data_vec.push_back(std::make_shared<arrow::ChunkedArray>(val_chunks));

    auto results = arrow::Table::Make(schema, data_vec);

    return results;
}

std::shared_ptr<arrow::Table> aggr_host(std::shared_ptr<arrow::Table> res) {
    auto options = std::make_shared<arrow::compute::ScalarAggregateOptions>();
    std::shared_ptr<ac::AggregateNodeOptions> aggregate_options;
    aggregate_options = std::make_shared<ac::AggregateNodeOptions>(ac::AggregateNodeOptions{{{"hash_sum", options, "order_count"}},
                                                                                             {"o_orderpriority"}});

    auto order_by_options = ac::OrderByNodeOptions({{arrow::compute::SortKey("o_orderpriority")}});

    auto plan = ac::Declaration::Sequence({{"table_source", ac::TableSourceNodeOptions(res)},
                                        {"aggregate", *aggregate_options},
                                        {"order_by", order_by_options}});

    ac::QueryOptions query_options;
    query_options.use_threads = true;
    auto result = ac::DeclarationToTable(plan, query_options);
    auto result_data = result.ValueOrDie();
 
    std::cout << result_data->ToString() << std::endl;

    return result_data;
}

double comp() {
    auto pred_l = cp::less(cp::field_ref("l_commitdate"),
                           cp::field_ref("l_receiptdate"));

    auto pred_o = cp::and_(cp::greater_equal(cp::field_ref("o_orderdate"),
                                             cp::literal(date_to_int("1993-08-01"))),
                           cp::less(cp::field_ref("o_orderdate"),
                                    cp::literal(date_to_int("1993-11-01"))));


    ac::Declaration o_in{"table_source", ac::TableSourceNodeOptions(orders)};
    ac::Declaration l_in{"table_source", ac::TableSourceNodeOptions(lineitem)};

    ac::FilterNodeOptions o_filt_options(pred_o);
    ac::Declaration o_filt{"filter", {std::move(o_in)}, o_filt_options};

    ac::FilterNodeOptions l_filt_options(pred_l);
    ac::Declaration l_filt{"filter", {std::move(l_in)}, l_filt_options};

    ac::HashJoinNodeOptions join_opts{
        ac::JoinType::INNER,
        {"o_orderkey"},
        {"l_orderkey"}, cp::literal(true), "_l", "_r"
    };

    ac::Declaration joined{
        "hashjoin", {std::move(o_filt), std::move(l_filt)}, std::move(join_opts)
    };

    auto un_options = std::make_shared<arrow::compute::ScalarAggregateOptions>();
    auto unique_options = ac::AggregateNodeOptions(ac::AggregateNodeOptions{{{"hash_distinct", un_options, "o_orderkey"}},
                                                                            {"o_orderkey", "o_orderpriority"}});

    ac::Declaration unique {"aggregate", {std::move(joined)}, std::move(unique_options)};

    auto options = std::make_shared<arrow::compute::CountOptions>();
    auto aggregate_options = ac::AggregateNodeOptions(ac::AggregateNodeOptions{{{"hash_count", options, "o_orderpriority", "order_count"}},
                                                                               {"o_orderpriority"}});

    ac::Declaration aggregate {"aggregate", {std::move(unique)}, std::move(aggregate_options)};

    auto order_by_options = ac::OrderByNodeOptions({{arrow::compute::SortKey("o_orderpriority")}});
    ac::Declaration ordered {"order_by", {std::move(aggregate)}, std::move(order_by_options)};

    ac::QueryOptions query_options;
    query_options.use_threads = true;

    auto query = ac::DeclarationToTable(ordered, query_options);
    auto table = query.ValueOrDie();
    std::cout << table->ToString() << std::endl;

    return 0;
}

int main(void) {
    dpu_set_t system;
    DPU_ASSERT(dpu_alloc(NR_DPU, "sgXferEnable=true", &system));
    std::vector<int32_t> l_cols = {0, 11, 12};
    auto status = parquet_to_table("lineitem", lineitem, l_cols);
    if (!status.ok()) {
        std::cout << status.message() << std::endl;
    }
    std::vector<int32_t> o_cols = {0, 4, 5};
    status = parquet_to_table("orders", orders, o_cols);
    if (!status.ok()) {
        std::cout << status.message() << std::endl;
    }
    try {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        {
            DPU_ASSERT(dpu_load(system, "kernel_q4_1", NULL));
            populate_mram_1(system);
            DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));

            {
                std::vector<std::vector<uint64_t>> sizes_o(NR_DPU, std::vector<uint64_t>(NR_DPU+1));
                get_vec(system, sizes_o, 2*524288*sizeof(key_ptr32), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);
                auto buf_o_orderkey = collect(system, 0, sizeof(key_ptr32));
                auto buf_o_orderprio = collect(system, 524288*sizeof(key_ptr32), 16);

                DPU_ASSERT(dpu_load(system, "kernel_q4_2", NULL));
                populate_mram_2(system);
                DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));
                std::vector<std::vector<uint64_t>> sizes_l(NR_DPU, std::vector<uint64_t>(NR_DPU+1));
                get_vec(system, sizes_l, 2*524288*sizeof(keyval_ptr32), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);
                auto buf_l_orderkey = collect(system, 0, sizeof(key_ptr32));

                DPU_ASSERT(dpu_load(system, "kernel_q4_3", NULL));
                distribute(system, sizes_o, sizes_l, buf_o_orderkey, buf_l_orderkey, buf_o_orderprio);
            }

            DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));

            DPU_ASSERT(dpu_load(system, "kernel_q4_4", NULL));
            DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));

            auto res = get_results(system);
            aggr_host(res);
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "Host elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " millisecs." << std::endl;

        //comp();
        output_dpu(system);
    }
    catch (const dpu::DpuError & e) {
        std::cerr << e.what() << std::endl;
    }
    
    return 0;
}