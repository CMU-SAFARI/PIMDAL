#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>

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

void populate_mram(dpu_set_t &system) {
    scatter_table(system, lineitem, "l_extendedprice", "l_extendedprice", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, lineitem, "l_discount", "l_discount", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, lineitem, "l_quantity", "l_quantity", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, lineitem, "l_tax", "l_tax", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, lineitem, "l_returnflag", "l_returnflag", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, lineitem, "l_linestatus", "l_linestatus", 0, DPU_SG_XFER_DEFAULT);
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

        query_args[dpu][0].date = date_to_int("1998-09-02");
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

arrow::ArrayVector get_res_char(dpu_set_t &system, uint32_t offset, uint32_t count,
                                 std::vector<std::vector<query_res_t>> query_res) {

    uint32_t length = (count + 7) & (-8);
    arrow::BufferVector buffers = alloc_buf_vec(length, NR_DPU);
    get_buf(system, buffers, offset, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);


    arrow::ArrayVector chunks;
    struct dpu_set_t dpu;
    uint32_t each_dpu;
    DPU_FOREACH(system, dpu, each_dpu) {
        uint32_t dpu_size = query_res[each_dpu][0].count;
        
        auto array_data = arrow::ArrayData::Make(
            arrow::fixed_size_binary(1),
            dpu_size,
            {nullptr, buffers[each_dpu]});
        auto array = arrow::MakeArray(array_data);
        chunks.push_back(array);
    }

    return chunks;
}

arrow::ArrayVector get_res_32(dpu_set_t &system, uint32_t offset, uint32_t count,
                               std::vector<std::vector<query_res_t>> query_res) {

    uint32_t length = (count*sizeof(uint32_t) + 7) & (-8);
    arrow::BufferVector buffers = alloc_buf_vec(length, NR_DPU);
    get_buf(system, buffers, offset, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);


    arrow::ArrayVector chunks;
    struct dpu_set_t dpu;
    uint32_t each_dpu;
    DPU_FOREACH(system, dpu, each_dpu) {
        uint32_t dpu_size = query_res[each_dpu][0].count;
        
        auto array_data = arrow::ArrayData::Make(
            arrow::int32(),
            dpu_size,
            {nullptr, buffers[each_dpu]});
        auto array = arrow::MakeArray(array_data);
        chunks.push_back(array);
    }

    return chunks;
}

arrow::ArrayVector get_res_64(dpu_set_t &system, uint32_t offset, uint32_t count,
                               std::vector<std::vector<query_res_t>> query_res) {

    uint32_t length = count*sizeof(int64_t);
    arrow::BufferVector buffers = alloc_buf_vec(length, NR_DPU);
    get_buf(system, buffers, offset, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    arrow::ArrayVector chunks;
    struct dpu_set_t dpu;
    uint32_t each_dpu;
    DPU_FOREACH(system, dpu, each_dpu) {
        uint32_t dpu_size = query_res[each_dpu][0].count;
        
        auto array_data = arrow::ArrayData::Make(
            arrow::int64(),
            dpu_size,
            {nullptr, buffers[each_dpu]});
        auto array = arrow::MakeArray(array_data);
        chunks.push_back(array);
    }

    return chunks;
}

std::shared_ptr<arrow::Table> get_results(dpu_set_t &system) {
    std::vector<std::vector<query_res_t>> query_res {NR_DPU, std::vector<query_res_t>(1)};

    get_vec(system, query_res, 0, "dpu_results", DPU_XFER_DEFAULT);

    arrow::ArrayVector l_returnflag_chunks;
    arrow::ArrayVector l_linestatus_chunks;
    arrow::ArrayVector sum_qty_chunks;
    arrow::ArrayVector sum_base_price_chunks;
    arrow::ArrayVector sum_disc_price_chunks;
    arrow::ArrayVector sum_charge_chunks;
    arrow::ArrayVector avg_disc_chunks;
    arrow::ArrayVector count_order_chunks;

    struct dpu_set_t dpu;
    uint32_t each_dpu;
    uint32_t max_count = 0;
    DPU_FOREACH(system, dpu, each_dpu) {
        //std::cout << "Size: " << gb_res[each_dpu][0].count << std::endl;
        if (query_res[each_dpu][0].count > max_count) {
            max_count = query_res[each_dpu][0].count;
        }
    }

    std::cout << "Count: " << max_count << std::endl;

    arrow::ChunkedArrayVector data_vec;
    auto schema = arrow::schema({arrow::field("l_returnflag", arrow::fixed_size_binary(1), false),
                                 arrow::field("l_linestatus", arrow::fixed_size_binary(1), false),
                                 arrow::field("sum_qty", arrow::int64(), false),
                                 arrow::field("sum_base_price", arrow::int64(), false),
                                 arrow::field("sum_disc_price", arrow::int64(), false),
                                 arrow::field("sum_charge", arrow::int64(), false),
                                 arrow::field("avg_disc", arrow::int64(), false),
                                 arrow::field("count_order", arrow::int32(), false)});

    uint32_t offset = 524288*sizeof(key_ptr32);
    data_vec.push_back(std::make_shared<arrow::ChunkedArray>(
        get_res_char(system, offset, max_count, query_res)
    ));

    offset += 16*sizeof(int64_t);
    data_vec.push_back(std::make_shared<arrow::ChunkedArray>(
        get_res_char(system, offset, max_count, query_res)
    ));

    offset += 16*sizeof(int64_t);
    data_vec.push_back(std::make_shared<arrow::ChunkedArray>(
        get_res_64(system, offset, max_count, query_res)
    ));

    offset += 16*sizeof(int64_t);
    data_vec.push_back(std::make_shared<arrow::ChunkedArray>(
        get_res_64(system, offset, max_count, query_res)
    ));

    offset += 16*sizeof(int64_t);
    data_vec.push_back(std::make_shared<arrow::ChunkedArray>(
        get_res_64(system, offset, max_count, query_res)
    ));

    offset += 16*sizeof(int64_t);
    data_vec.push_back(std::make_shared<arrow::ChunkedArray>(
        get_res_64(system, offset, max_count, query_res)
    ));

    offset += 16*sizeof(int64_t);
    data_vec.push_back(std::make_shared<arrow::ChunkedArray>(
        get_res_64(system, offset, max_count, query_res)
    ));

    offset += 16*sizeof(int64_t);
    data_vec.push_back(std::make_shared<arrow::ChunkedArray>(
        get_res_32(system, offset, max_count, query_res)
    ));

    return arrow::Table::Make(schema, data_vec);
}

std::shared_ptr<arrow::ChunkedArray> aggr_host(std::shared_ptr<arrow::Table> res) {

    auto aggregate_options =
    ac::AggregateNodeOptions{{{"hash_sum", nullptr, "sum_qty", "sum_qty"},
                            {"hash_sum", nullptr, "sum_base_price", "sum_base_price"},
                            {"hash_sum", nullptr, "sum_disc_price", "sum_disc_price"},
                            {"hash_sum", nullptr, "sum_charge", "sum_charge"},
                            {"hash_sum", nullptr, "avg_disc", "avg_disc"},
                            {"hash_sum", nullptr, "count_order", "count_order"}},
                            {"l_returnflag", "l_linestatus"}};

    arrow::compute::Expression l_returnflag = cp::field_ref("l_returnflag");
    arrow::compute::Expression l_linestatus = cp::field_ref("l_linestatus");
    arrow::compute::Expression sum_qty = cp::field_ref("sum_qty");
    arrow::compute::Expression sum_base_price = cp::field_ref("sum_base_price");
    arrow::compute::Expression sum_disc_price = cp::field_ref("sum_disc_price");
    arrow::compute::Expression sum_charge = cp::field_ref("sum_charge");
    arrow::compute::Expression avg_qty = cp::call("divide", {cp::field_ref("sum_qty"), cp::field_ref("count_order")});
    arrow::compute::Expression avg_price = cp::call("divide", {cp::field_ref("sum_base_price"), cp::field_ref("count_order")});
    arrow::compute::Expression avg_disc = cp::call("divide", {cp::field_ref("avg_disc"), cp::field_ref("count_order")});
    arrow::compute::Expression count_order = cp::field_ref("count_order");

    std::vector<arrow::compute::Expression> projection_list = {
        l_returnflag, l_linestatus, sum_qty, sum_base_price, sum_disc_price, sum_charge,
        avg_qty, avg_price, avg_disc, count_order
    };
    std::vector<std::string> projection_names = {
        "l_returnflag", "l_linestatus", "sum_qty", "sum_base_price", "sum_disc_price",
        "sum_charge", "avg_qty", "avg_price", "avg_disc", "count_order"
    };
    ac::ProjectNodeOptions project_opts(std::move(projection_list), std::move(projection_names));

    ac::Declaration plan = ac::Declaration::Sequence({{"table_source", ac::TableSourceNodeOptions(res)},
                                                      {"aggregate", aggregate_options},
                                                      {"project", project_opts}});

    ac::QueryOptions query_options;
    query_options.use_threads = true;

    auto query = ac::DeclarationToTable(plan, query_options);
    auto table = query.ValueOrDie();
    std::cout << table->ToString() << std::endl;

    return table->GetColumnByName("revenue");
}

void comp() {

    auto pred_date = cp::less(cp::field_ref("l_shipdate"),
                              cp::literal(date_to_int("1998-09-02")));

    arrow::compute::Expression l_returnflag = cp::field_ref("l_returnflag");
    arrow::compute::Expression l_linestatus = cp::field_ref("l_linestatus");
    arrow::compute::Expression sum_qty = cp::field_ref("l_quantity");
    arrow::compute::Expression sum_base_price = cp::field_ref("l_extendedprice");
    arrow::compute::Expression sum_disc_price = cp::call(
        "multiply", {cp::field_ref("l_extendedprice"),
                     cp::call("subtract", {cp::literal(100), cp::field_ref("l_discount")})}
    );
    arrow::compute::Expression sum_charge = cp::call(
        "multiply", {cp::field_ref("l_extendedprice"),
                        cp::call("multiply", {
                            cp::call("subtract", {
                                cp::literal(100),
                                cp::field_ref("l_discount")
                            }),
                            cp::call("add", {
                                cp::literal(100),
                                cp::field_ref("l_tax")
                            })
                        })
                    }
    );
    arrow::compute::Expression avg_qty = cp::field_ref("l_quantity");
    arrow::compute::Expression avg_price = cp::field_ref("l_extendedprice");
    arrow::compute::Expression avg_disc = cp::field_ref("l_discount");
    arrow::compute::Expression count_order = cp::field_ref("l_orderkey");

    std::vector<arrow::compute::Expression> projection_list = {
        l_returnflag, l_linestatus, sum_qty, sum_base_price, sum_disc_price, sum_charge,
        avg_qty, avg_price, avg_disc, count_order
    };
    std::vector<std::string> projection_names = {
        "l_returnflag", "l_linestatus", "sum_qty", "sum_base_price", "sum_disc_price",
        "sum_charge", "avg_qty", "avg_price", "avg_disc", "count_order"
    };
    ac::ProjectNodeOptions project_opts(std::move(projection_list), std::move(projection_names));

    auto aggregate_options =
        ac::AggregateNodeOptions{{{"hash_sum", nullptr, "sum_qty", "sum_qty"},
                                  {"hash_sum", nullptr, "sum_base_price", "sum_base_price"},
                                  {"hash_sum", nullptr, "sum_disc_price", "sum_disc_price"},
                                  {"hash_sum", nullptr, "sum_charge", "sum_charge"},
                                  {"hash_mean", nullptr, "avg_qty", "avg_qty"},
                                  {"hash_mean", nullptr, "avg_price", "avg_price"},
                                  {"hash_mean", nullptr, "avg_disc", "avg_disc"},
                                  {"hash_count", nullptr, "count_order", "count_order"}},
                                 {"l_returnflag", "l_linestatus"}};

    ac::Declaration plan = ac::Declaration::Sequence({{"table_source", ac::TableSourceNodeOptions(lineitem)},
                                                      {"filter", ac::FilterNodeOptions(pred_date)},
                                                      {"project", project_opts},
                                                      {"aggregate", aggregate_options}});

    ac::QueryOptions query_options;
    query_options.use_threads = true;

    auto query = ac::DeclarationToTable(plan, query_options);
    auto table = query.ValueOrDie();
    std::cout << table->ToString() << std::endl;
}

int main(void) {
    dpu_set_t system;
    DPU_ASSERT(dpu_alloc(NR_DPU, "sgXferEnable=true", &system));
    std::vector<int32_t> sel_cols = {0, 4, 5, 6, 7, 8, 9, 10};
    auto status = parquet_to_table("lineitem", lineitem, sel_cols);
    if (!status.ok()) {
        std::cout << status.message() << std::endl;
    }
    try {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        DPU_ASSERT(dpu_load(system, "kernel_q1_1", NULL));
        populate_mram(system);
        DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));

        DPU_ASSERT(dpu_load(system, "kernel_q1_2", NULL));
        DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));

        auto res = get_results(system);
        
        aggr_host(res);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "Host elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " millisecs." << std::endl;

        comp();
        output_dpu(system);
    }
    catch (const dpu::DpuError & e) {
        std::cerr << e.what() << std::endl;
    }
    
    return 0;
}