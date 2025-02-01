#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>

#include <arrow/api.h>
#include <arrow/acero/exec_plan.h>
#include <arrow/dataset/api.h>
#include "../../reader/read_table.cpp"

#ifndef NR_DPU
#define NR_DPU 4
#endif

#include "transfer_helper.h"

#include "param.h"
#include "datatype.h"

namespace ac = arrow::acero;
namespace cp = arrow::compute;

extern std::shared_ptr<arrow::Table> lineitem;

void populate_mram(dpu_set_t &system) {
    scatter_table(system, lineitem, "l_quantity", "l_quantity", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, lineitem, "l_extendedprice", "l_extendedprice", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, lineitem, "l_discount", "l_discount", 0, DPU_SG_XFER_DEFAULT);
    scatter_table(system, lineitem, "l_shipdate", "l_shipdate", 0, DPU_SG_XFER_DEFAULT);

    std::vector<std::vector<query_args_t>> query_args {NR_DPU, std::vector<query_args_t>(1)};
    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        if (lineitem->num_rows() % NR_DPU == 0) {
            query_args[dpu][0].size = lineitem->num_rows()/NR_DPU;
        }
        else {
            if (dpu < NR_DPU - 1) {
                query_args[dpu][0].size = lineitem->num_rows()/NR_DPU + 1;
            }
            else {
                query_args[dpu][0].size = lineitem->num_rows() % (lineitem->num_rows()/NR_DPU + 1);
            }
        }

        query_args[dpu][0].date_start = date_to_int("1994-01-01");
        query_args[dpu][0].date_end = date_to_int("1995-01-01");
        query_args[dpu][0].discount = 6;
        query_args[dpu][0].quantity = 24;
    }

    dist_vec(system, query_args, 0, "dpu_args", DPU_XFER_DEFAULT);
}

double get_result(dpu_set_t &system) {
    std::vector<std::vector<query_res_t>> query_res {NR_DPU, std::vector<query_res_t>(1)};

    get_vec(system, query_res, 0, "dpu_results", DPU_XFER_DEFAULT);

    double revenue = 0;
    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        //std::cout << "Revenue dpu: " << query_res[dpu][0].revenue << std::endl;
        revenue += query_res[dpu][0].revenue;
    }

    return revenue / 10000;
}

std::shared_ptr<arrow::ChunkedArray> comp() {
    auto pred_date = cp::and_(cp::greater_equal(cp::field_ref("l_shipdate"),
                                                cp::literal(date_to_int("1994-01-01"))),
                              cp::less(cp::field_ref("l_shipdate"),
                                       cp::literal(date_to_int("1995-01-01"))));

    auto pred_quantity = cp::less(cp::field_ref("l_quantity"),
                                  cp::literal(24));

    auto pred_discount = cp::and_(cp::greater_equal(cp::field_ref("l_discount"),
                                                    cp::literal(5)),
                                  cp::less_equal(cp::field_ref("l_discount"),
                                                 cp::literal(7)));

    arrow::compute::Expression revenue = cp::call("multiply", {cp::field_ref("l_discount"), cp::field_ref("l_extendedprice")});
    std::vector<arrow::compute::Expression> projection_list = {revenue};
    std::vector<std::string> projection_names = {"revenue"};
    ac::ProjectNodeOptions project_opts(std::move(projection_list), std::move(projection_names));

    auto aggregate_options =
      ac::AggregateNodeOptions{{{"sum", nullptr, "revenue", "revenue"}}};

    ac::Declaration plan = ac::Declaration::Sequence({{"table_source", ac::TableSourceNodeOptions(lineitem)},
                                                      {"filter", ac::FilterNodeOptions(pred_date)},
                                                      {"filter", ac::FilterNodeOptions(pred_quantity)},
                                                      {"filter", ac::FilterNodeOptions(pred_discount)},
                                                      {"project", project_opts},
                                                      {"aggregate", aggregate_options}});

    ac::QueryOptions query_options;
    query_options.use_threads = true;

    auto query = ac::DeclarationToTable(plan, query_options);
    auto table = query.ValueOrDie();
    std::cout << table->ToString() << std::endl;

    return table->GetColumnByName("revenue");
}

std::shared_ptr<arrow::Table> lineitem;

int main(void) {
    dpu_set_t system;
    DPU_ASSERT(dpu_alloc(NR_DPU, "sgXferEnable=true", &system));
    std::vector<int32_t> sel_cols = {4, 5, 6, 10};
    auto status = parquet_to_table("lineitem", lineitem, sel_cols);
    if (!status.ok()) {
        std::cout << status.message() << std::endl;
    }
    try {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        DPU_ASSERT(dpu_load(system, "kernel_q6", NULL));
        populate_mram(system);
        DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));
        double revenue = get_result(system);
        std::cout.precision(11);
        std::cout << "Revenue: " << revenue << std::endl;

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "Host elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " millisecs." << std::endl;

        auto comparsion = comp();
        output_dpu(system);
    }
    catch (const dpu::DpuError & e) {
        std::cerr << e.what() << std::endl;
    }
    
    return 0;
}