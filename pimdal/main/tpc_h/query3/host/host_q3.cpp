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

void populate_mram_1(dpu_set_t &system) {
    scatter_table(system, customer, "c_custkey", "c_custkey", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, customer, "c_mktsegment", "c_mktsegment", 0, DPU_SG_XFER_DEFAULT);

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

        strcpy(query_args[dpu][0].c_segment, "BUILDING");
    }

    dist_vec(system, query_args, 0, "dpu_args", DPU_XFER_DEFAULT);
}

void populate_mram_2(dpu_set_t &system) {
    scatter_table(system, orders, "o_orderkey", "o_orderkey", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, orders, "o_custkey", "o_custkey", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, orders, "o_orderdate", "o_orderdate", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, orders, "o_shippriority", "o_shippriority", 0, DPU_SG_XFER_DEFAULT);

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

        query_args[dpu][0].o_date = date_to_int("1995-03-15");
    }

    dist_vec(system, query_args, 0, "dpu_args", DPU_XFER_DEFAULT);
}

void populate_mram_3(dpu_set_t &system) {
    scatter_table(system, lineitem, "l_extendedprice", "l_extendedprice", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, lineitem, "l_discount", "l_discount", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, lineitem, "l_orderkey", "l_orderkey", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, lineitem, "l_shipdate", "l_shipdate", 0, DPU_SG_XFER_DEFAULT);

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

        query_args[dpu][0].l_date = date_to_int("1995-03-15");
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
    std::cout << "Total count: " << total << " max: " << max_sel_size << std::endl;

    uint32_t size_rounded = (max_sel_size*type_size + 7) & (-8);
    arrow::BufferVector buffers = alloc_buf_vec(size_rounded, NR_DPU);
    get_buf(system, buffers, offset, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    return buffers;
}

void distribute_1(dpu_set_t &system, std::vector<std::vector<uint64_t>> sizes_c,
                  std::vector<std::vector<uint64_t>> sizes_o,
                  arrow::BufferVector buf_c_custkey, arrow::BufferVector buf_o_custkey,
                  arrow::BufferVector buf_o_orderkey, arrow::BufferVector buf_o_orderdate,
                  arrow::BufferVector buf_o_shipprio) {
    
    std::vector<std::vector<query_args_t>> dpu_args (NR_DPU, std::vector<query_args_t>(1));
    std::vector<std::vector<uint64_t>> off_data (NR_DPU, std::vector<uint64_t>(NR_DPU+1));

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

    // Distributing o_orderkey
    sg_xfer_context_2d sc_args_orderkey = {.partitions = buf_o_orderkey, .offset = off_data};
    get_block_t get_block_info_orderkey = {.f = &get_cpy_ptr_2d, .args = &sc_args_orderkey, .args_size=(sizeof(sc_args_orderkey))};

    uint32_t length = (max_outer_size*sizeof(uint32_t) + 7) & (-8);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 2*524288*sizeof(key_ptr32),
                                length, &get_block_info_orderkey, flag));

    // Distributing o_orderdate
    sg_xfer_context_2d sc_args_orderdate = {.partitions = buf_o_orderdate, .offset = off_data};
    get_block_t get_block_info_orderdate = {.f = &get_cpy_ptr_2d, .args = &sc_args_orderdate, .args_size=(sizeof(sc_args_orderdate))};

    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 3*524288*sizeof(key_ptr32),
                                length, &get_block_info_orderdate, flag));

    // Distributing o_shipprio
    sg_xfer_context_2d sc_args_shipprio = {.partitions = buf_o_shipprio, .offset = off_data};
    get_block_t get_block_info_shipprio = {.f = &get_cpy_ptr_2d, .args = &sc_args_shipprio, .args_size=(sizeof(sc_args_shipprio))};

    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 4*524288*sizeof(key_ptr32),
                                length, &get_block_info_shipprio, flag));

    dist_vec(system, dpu_args, 0, "dpu_args", DPU_XFER_DEFAULT);

}

void distribute_2(dpu_set_t &system, std::vector<std::vector<uint64_t>> sizes_o,
                  std::vector<std::vector<uint64_t>> sizes_l,
                  arrow::BufferVector buf_o_orderkey, arrow::BufferVector buf_l_orderkey,
                  arrow::BufferVector buf_o_orderdate, arrow::BufferVector buf_o_shipprio,
                  arrow::BufferVector buf_l_extendedprice, arrow::BufferVector buf_l_discount) {
    
    std::vector<std::vector<query_args_t>> dpu_args (NR_DPU, std::vector<query_args_t>(1));
    std::vector<std::vector<uint64_t>> off_data_o (NR_DPU, std::vector<uint64_t>(NR_DPU+1));
    std::vector<std::vector<uint64_t>> off_data_l (NR_DPU, std::vector<uint64_t>(NR_DPU+1));

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
            off_data_o[dpu][i] = sizes_o[dpu][i] * sizeof(uint32_t);
            off_data_l[dpu][i] = sizes_l[dpu][i] * sizeof(int64_t);
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

    // Distributing o_orderdate
    sg_xfer_context_2d sc_args_orderdate = {.partitions = buf_o_orderdate, .offset = off_data_o};
    get_block_t get_block_info_orderdate = {.f = &get_cpy_ptr_2d, .args = &sc_args_orderdate, .args_size=(sizeof(sc_args_orderdate))};

    uint32_t length = (max_inner_size*sizeof(uint32_t) + 7) & (-8);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 2*524288*sizeof(key_ptr32),
                                length, &get_block_info_orderdate, flag));

    // Distributing o_shipprio
    sg_xfer_context_2d sc_args_shipprio = {.partitions = buf_o_shipprio, .offset = off_data_o};
    get_block_t get_block_info_shipprio = {.f = &get_cpy_ptr_2d, .args = &sc_args_shipprio, .args_size=(sizeof(sc_args_shipprio))};

    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 3*524288*sizeof(key_ptr32),
                                length, &get_block_info_shipprio, flag));

    // Distributing l_extendedprice
    sg_xfer_context_2d sc_args_extendedprice = {.partitions = buf_l_extendedprice, .offset = off_data_l};
    get_block_t get_block_info_extendedprice = {.f = &get_cpy_ptr_2d, .args = &sc_args_extendedprice, .args_size=(sizeof(sc_args_extendedprice))};

    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 4*524288*sizeof(key_ptr32),
                                max_outer_size*sizeof(int64_t), &get_block_info_extendedprice, flag));

    // Distributing l_discount
    sg_xfer_context_2d sc_args_discount = {.partitions = buf_l_discount, .offset = off_data_l};
    get_block_t get_block_info_discount = {.f = &get_cpy_ptr_2d, .args = &sc_args_discount, .args_size=(sizeof(sc_args_discount))};

    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 5*524288*sizeof(key_ptr32),
                                max_outer_size*sizeof(int64_t), &get_block_info_discount, flag));

    dist_vec(system, dpu_args, 0, "dpu_args", DPU_XFER_DEFAULT);

}

std::shared_ptr<arrow::Table> get_results(dpu_set_t &system) {

    arrow::BufferVector buffers_key = alloc_buf_vec(16*sizeof(uint32_t), NR_DPU);
    get_buf(system, buffers_key, 0, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    arrow::BufferVector buffers_rev = alloc_buf_vec(16*sizeof(int64_t), NR_DPU);
    get_buf(system, buffers_rev, 16*sizeof(key_ptr32), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    arrow::BufferVector buffers_date = alloc_buf_vec(16*sizeof(uint32_t), NR_DPU);
    get_buf(system, buffers_date, 32*sizeof(key_ptr32), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    arrow::BufferVector buffers_prio = alloc_buf_vec(16*sizeof(uint32_t), NR_DPU);
    get_buf(system, buffers_prio, 48*sizeof(key_ptr32), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    arrow::Result<std::unique_ptr<arrow::Buffer>> res_key_try = arrow::AllocateBuffer(10*sizeof(uint32_t));
    std::shared_ptr<arrow::Buffer> res_key = *std::move(res_key_try);
    uint32_t* res_key_data = (uint32_t*) res_key->mutable_data();

    arrow::Result<std::unique_ptr<arrow::Buffer>> res_rev_try = arrow::AllocateBuffer(10*sizeof(int64_t));
    std::shared_ptr<arrow::Buffer> res_rev = *std::move(res_rev_try);
    int64_t* res_rev_data = (int64_t*) res_rev->mutable_data();

    arrow::Result<std::unique_ptr<arrow::Buffer>> res_date_try = arrow::AllocateBuffer(10*sizeof(uint32_t));
    std::shared_ptr<arrow::Buffer> res_date = *std::move(res_date_try);
    uint32_t* res_date_data = (uint32_t*) res_date->mutable_data();

    arrow::Result<std::unique_ptr<arrow::Buffer>> res_prio_try = arrow::AllocateBuffer(10*sizeof(uint32_t));
    std::shared_ptr<arrow::Buffer> res_prio = *std::move(res_prio_try);
    uint32_t* res_prio_data = (uint32_t*) res_prio->mutable_data();

    uint32_t indices[NR_DPU] = {0};
    for (uint32_t i = 0; i < 10; i++) {
        int64_t max_rev = 0;
        uint32_t max_dpu = 0;
        for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
            const int64_t* data = (const int64_t*) buffers_rev[dpu]->data();
            if (data[indices[dpu]] > max_rev) {
                max_rev = data[indices[dpu]];
                max_dpu = dpu;
            }
        }

        const uint32_t* key_data = (const uint32_t*) buffers_key[max_dpu]->data();
        res_key_data[i] = key_data[indices[max_dpu]];

        res_rev_data[i] = max_rev;

        const uint32_t* date_data = (const uint32_t*) buffers_date[max_dpu]->data();
        res_date_data[i] = date_data[indices[max_dpu]];

        const uint32_t* prio_data = (const uint32_t*) buffers_prio[max_dpu]->data();
        res_prio_data[i] = prio_data[indices[max_dpu]];

        indices[max_dpu]++;
    }

    auto array_data_key = arrow::ArrayData::Make(arrow::uint32(), 10, {nullptr, res_key});
    auto array_key = arrow::MakeArray(array_data_key);

    auto array_data_rev = arrow::ArrayData::Make(arrow::int64(), 10, {nullptr, res_rev});
    auto array_rev = arrow::MakeArray(array_data_rev);

    auto array_data_date = arrow::ArrayData::Make(arrow::uint32(), 10, {nullptr, res_date});
    auto array_date = arrow::MakeArray(array_data_date);

    auto array_data_prio = arrow::ArrayData::Make(arrow::uint32(), 10, {nullptr, res_prio});
    auto array_prio = arrow::MakeArray(array_data_prio);

    arrow::ArrayVector data_vec;
    auto schema = arrow::schema({arrow::field("l_orderkey", arrow::uint32(), false),
                                 arrow::field("revenue", arrow::int64(), false),
                                 arrow::field("o_orderdate", arrow::uint32(), false),
                                 arrow::field("o_shippriority", arrow::uint32(), false)});
    data_vec.push_back(array_key);
    data_vec.push_back(array_rev);
    data_vec.push_back(array_date);
    data_vec.push_back(array_prio);

    auto results = arrow::Table::Make(schema, data_vec);

    return results;
}

double comp() {
    auto pred_l = cp::less(cp::field_ref("l_shipdate"),
                           cp::literal(date_to_int("1995-03-15")));

    auto pred_o = cp::less(cp::field_ref("o_orderdate"),
                           cp::literal(date_to_int("1995-03-15")));

    auto pred_c = cp::equal(cp::field_ref("c_maktsegment"),
                            cp::literal("4255494C44494E4700000000000000"));

    ac::Declaration l_in{"table_source", ac::TableSourceNodeOptions(lineitem)};
    ac::Declaration o_in{"table_source", ac::TableSourceNodeOptions(orders)};
    ac::Declaration c_in{"table_source", ac::TableSourceNodeOptions(customer)};

    ac::FilterNodeOptions l_filt_options(pred_l);
    ac::Declaration l_filt{"filter", {std::move(l_in)}, l_filt_options};

    ac::FilterNodeOptions o_filt_options(pred_o);
    ac::Declaration o_filt{"filter", {std::move(o_in)}, o_filt_options};

    ac::FilterNodeOptions c_filt_options(pred_c);
    ac::Declaration c_filt{"filter", {std::move(c_in)}, c_filt_options};

    ac::HashJoinNodeOptions join_opts_1{
        ac::JoinType::INNER,
        {"o_orderkey"},
        {"l_orderkey"}, cp::literal(true), "_l", "_r"
    };

    ac::Declaration joined_1{
        "hashjoin", {std::move(o_filt), std::move(l_filt)}, std::move(join_opts_1)
    };

    ac::HashJoinNodeOptions join_opts_2{
        ac::JoinType::INNER,
        {"c_custkey"},
        {"o_custkey"}, cp::literal(true), "_l", "_r"
    };

    ac::Declaration joined_2{
        "hashjoin", {std::move(c_filt), std::move(joined_1)}, std::move(join_opts_2)
    };

    cp::Expression l_orderkey = cp::field_ref("l_orderkey");
    cp::Expression revenue = cp::call(
        "multiply", {cp::field_ref("l_extendedprice"),
                     cp::call("subtract", {cp::literal(100), cp::field_ref("l_discount")})}
    );
    cp::Expression o_orderdate = cp::field_ref("o_orderdate");
    cp::Expression o_shippriority = cp::field_ref("o_shippriority");

    std::vector<cp::Expression> projection_list = {
        l_orderkey, revenue, o_orderdate, o_shippriority
    };
    std::vector<std::string> projection_names = {
        "l_orderkey", "revenue", "o_orderdate", "o_shippriority"
    };
    ac::ProjectNodeOptions project_opts(std::move(projection_list), std::move(projection_names));

    ac::Declaration rev {"projection", {std::move(joined_2)}, std::move(project_opts)};

    auto sum_opt = std::make_shared<arrow::compute::ScalarAggregateOptions>();
    auto sum_options = ac::AggregateNodeOptions(ac::AggregateNodeOptions{{{"hash_sum", sum_opt, "l_orderkey", "revenue"}},
                                                                         {"l_orderkey", "o_orderdate", "o_shippriority"}});

    ac::Declaration summed {"aggregate", {std::move(rev)}, std::move(sum_options)};

    ac::QueryOptions query_options;
    query_options.use_threads = true;

    auto query = ac::DeclarationToTable(summed, query_options);
    auto table = query.ValueOrDie();
    std::cout << table->ToString() << std::endl;

    return 0;
}

int main(void) {
    dpu_set_t system;
    DPU_ASSERT(dpu_alloc(NR_DPU, "sgXferEnable=true", &system));
    std::vector<int32_t> c_cols = {0, 6};
    auto status = parquet_to_table("customer", customer, c_cols);
    if (!status.ok()) {
        std::cout << status.message() << std::endl;
    }

    std::vector<int32_t> o_cols = {0, 1, 4, 7};
    status = parquet_to_table("orders", orders, o_cols);
    if (!status.ok()) {
        std::cout << status.message() << std::endl;
    }

    std::vector<int32_t> l_cols = {0, 5, 6, 10};
    status = parquet_to_table("lineitem", lineitem, l_cols);
    if (!status.ok()) {
        std::cout << status.message() << std::endl;
    }
    try {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        {
            DPU_ASSERT(dpu_load(system, "kernel_q3_1", NULL));
            populate_mram_1(system);
            DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));

            {
                std::vector<std::vector<uint64_t>> sizes_c(NR_DPU, std::vector<uint64_t>(NR_DPU+1));
                get_vec(system, sizes_c, 2*524288*sizeof(key_ptrtext), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);
                auto buf_c_custkey = collect(system, 0, sizeof(key_ptr32));

                DPU_ASSERT(dpu_load(system, "kernel_q3_2", NULL));
                populate_mram_2(system);
                DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));
                std::vector<std::vector<uint64_t>> sizes_o(NR_DPU, std::vector<uint64_t>(NR_DPU+1));
                get_vec(system, sizes_o, 4*524288*sizeof(key_ptr32), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);
                auto buf_o_custkey = collect(system, 0, sizeof(key_ptr32));
                auto buf_o_orderkey = collect(system, 524288*sizeof(key_ptr32), sizeof(uint32_t));
                auto buf_o_orderdate = collect(system, 2*524288*sizeof(key_ptr32), sizeof(uint32_t));
                auto buf_o_shipprio = collect(system, 3*524288*sizeof(key_ptr32), sizeof(uint32_t));

                DPU_ASSERT(dpu_load(system, "kernel_q3_3", NULL));
                distribute_1(system, sizes_c, sizes_o, buf_c_custkey, buf_o_custkey,
                            buf_o_orderkey, buf_o_orderdate, buf_o_shipprio);
            }

            {
                DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));
                std::vector<std::vector<uint64_t>> sizes_o(NR_DPU, std::vector<uint64_t>(NR_DPU+1));
                get_vec(system, sizes_o, 7*524288*sizeof(key_ptr32), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);
                auto buf_o_orderkey = collect(system, 0, sizeof(key_ptr32));
                auto buf_o_orderdate = collect(system, 524288*sizeof(key_ptr32), sizeof(uint32_t));
                auto buf_o_shipprio = collect(system, 2*524288*sizeof(key_ptr32), sizeof(uint32_t));

                DPU_ASSERT(dpu_load(system, "kernel_q3_4", NULL));
                populate_mram_3(system);
                DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));
                std::vector<std::vector<uint64_t>> sizes_l(NR_DPU, std::vector<uint64_t>(NR_DPU+1));
                get_vec(system, sizes_l, 3*524288*sizeof(key_ptr32), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);
                auto buf_l_orderkey = collect(system, 0, sizeof(key_ptr32));
                auto buf_l_extendedprice = collect(system, 524288*sizeof(key_ptr32), sizeof(int64_t));
                auto buf_l_discount = collect(system, 2*524288*sizeof(key_ptr32), sizeof(int64_t));

                DPU_ASSERT(dpu_load(system, "kernel_q3_5", NULL));
                distribute_2(system, sizes_o, sizes_l, buf_o_orderkey, buf_l_orderkey,
                            buf_o_orderdate, buf_o_shipprio, buf_l_extendedprice, buf_l_discount);
            }

            DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));

            auto res = get_results(system);
            std::cout << res->ToString() << std::endl;
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "Host elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " millisecs." << std::endl;

        //output_dpu(system);
    }
    catch (const dpu::DpuError & e) {
        std::cerr << e.what() << std::endl;
    }
    
    return 0;
}