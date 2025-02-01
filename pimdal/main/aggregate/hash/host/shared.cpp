#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <arrow/api.h>
#include <arrow/compute/api_aggregate.h>
#include <arrow/acero/exec_plan.h>
#include <arrow/dataset/api.h>

#include "transfer_helper.h"
#include "datatype.h"
#include "groupby.h"

#ifndef NR_DPU
#define NR_DPU 4
#endif
#ifndef BUFFER_SIZE
#define BUFFER_SIZE 400000
#endif
#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

namespace ac = arrow::acero;

const uint32_t aggr_type = COUNT;

std::shared_ptr<arrow::Table> table;

void init_buffer() {

    std::shared_ptr<arrow::Buffer> buffer_key;
    arrow::Result<std::unique_ptr<arrow::Buffer>> buffer_try = arrow::AllocateBuffer((uint64_t) NR_DPU*BUFFER_SIZE*sizeof(int32_t));
    if (!buffer_try.ok()) {
        std::cout << "Could not allocate buffer!" << std::endl;
    }
    buffer_key = *std::move(buffer_try);

    int32_t* buffer_data = (int32_t*) buffer_key->mutable_data();
    uint64_t buffer_size = buffer_key->size() / sizeof(int32_t);

    #pragma omp parallel
    {
        uint32_t seed = 123273;
    #ifdef _OPENMP
        seed += omp_get_thread_num();
    #endif
        std::default_random_engine gen(seed);
        std::uniform_int_distribution<int32_t> dist(1, 4000);

    #pragma omp for schedule(static)
        for(uint64_t j = 0; j < buffer_size; j++) {
            buffer_data[j] = dist(gen);
        }
    }

    auto schema = arrow::schema({arrow::field("key", arrow::int32(), false)});

    auto key_data = arrow::ArrayData::Make(arrow::int32(), NR_DPU*BUFFER_SIZE, {nullptr, buffer_key});
    auto key_array = arrow::MakeArray(key_data);

    arrow::ArrayVector data_vec;
    data_vec.push_back(key_array);

    if (aggr_type > 0) {
        std::shared_ptr<arrow::Buffer> buffer_val;
        arrow::Result<std::unique_ptr<arrow::Buffer>> buffer_val_try = arrow::AllocateBuffer((uint64_t) NR_DPU*BUFFER_SIZE*sizeof(int32_t));
        if (!buffer_val_try.ok()) {
            std::cout << "Could not allocate buffer!" << std::endl;
        }
        buffer_val = *std::move(buffer_val_try);

        buffer_data = (int32_t*) buffer_val->mutable_data();
        buffer_size = buffer_val->size() / sizeof(int32_t);

        #pragma omp parallel
        {
            uint32_t seed = 0;
        #ifdef _OPENMP
            seed += omp_get_thread_num();
        #endif
            std::default_random_engine gen(seed);
            std::uniform_int_distribution<int32_t> dist(1, 10);

        #pragma omp for
            for(uint64_t j = 0; j < buffer_size; j++) {
                buffer_data[j] = dist(gen);
            }
        }

        schema = arrow::schema({arrow::field("key", arrow::int32(), false),
                                arrow::field("val", arrow::int32(), false)});

        auto val_data = arrow::ArrayData::Make(arrow::int32(), NR_DPU*BUFFER_SIZE, {nullptr, buffer_key});
        auto val_array = arrow::MakeArray(val_data);

        data_vec.push_back(key_array);
    }

    table = arrow::Table::Make(schema, data_vec);
}

void validate(std::shared_ptr<arrow::Table> results) {

    std::shared_ptr<ac::AggregateNodeOptions> aggregate_options;
    if (aggr_type == SUM) {
        auto options = std::make_shared<arrow::compute::ScalarAggregateOptions>();
        aggregate_options = std::make_shared<ac::AggregateNodeOptions>(ac::AggregateNodeOptions{{{"hash_sum", options, "val", "sum(val)"}},
                                                                                                {"key"}});
    }
    else if (aggr_type == COUNT) {
        auto options = std::make_shared<arrow::compute::CountOptions>();
        aggregate_options = std::make_shared<ac::AggregateNodeOptions>(ac::AggregateNodeOptions{{{"hash_count", options, "key", "count(key)"}},
                                                                                                {"key"}});
    }
    else {
        auto options = std::make_shared<arrow::compute::ScalarAggregateOptions>();
        aggregate_options = std::make_shared<ac::AggregateNodeOptions>(ac::AggregateNodeOptions{{{"hash_distinct", options, "key"}},
                                                                                                {"key"}});
    }

    auto order_by_options = ac::OrderByNodeOptions({{arrow::compute::SortKey("key")}});
    auto plan = ac::Declaration::Sequence({{"table_source", ac::TableSourceNodeOptions(table)},
                                        {"aggregate", *aggregate_options},
                                        {"order_by", order_by_options}});

    ac::QueryOptions query_options;
    query_options.use_threads = true;

    std::chrono::steady_clock::time_point begin_comp = std::chrono::steady_clock::now();
    auto comp = ac::DeclarationToTable(plan, query_options);
    auto comp_table = comp.ValueOrDie();
    std::chrono::steady_clock::time_point end_comp = std::chrono::steady_clock::now();

    std::cout << "Comparison elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_comp - begin_comp).count()
              << " millisecs." << std::endl;

    //std::cout << results->ToString() << std::endl;
    //std::cout << "Length: " << results->num_rows() << std::endl;
    //std::cout << comp_table->ToString() << std::endl;
    //std::cout << "Length: " << comp_table->num_rows() << std::endl;
    if (verify_table(results, comp_table)) {
        std::cout << "No error!" << std::endl;
    }
    else {
        std::cout << "Incorrect result" << std::endl;
    }
}

void output_perf(dpu_set_t &system) {
    std::vector<std::vector<uint64_t>> cycles(NR_DPU, std::vector<uint64_t>(1));
    get_vec(system, cycles, 0, "cycles", DPU_XFER_DEFAULT);

#if PERF == 1
    std::ofstream file_cycles("haggr_cycles.csv", std::ofstream::app);
    std::ofstream file_time("haggr_time.csv", std::ofstream::app);

    file_cycles << NR_TASKLETS;
    file_time << NR_TASKLETS;

    std::vector<std::vector<uint32_t>> clocks_sec(NR_DPU, std::vector<uint32_t>(1));
    get_vec(system, clocks_sec, 0, "CLOCKS_PER_SEC", DPU_XFER_DEFAULT);
#else
    std::ofstream file_inst("haggr_inst.csv", std::ofstream::app);

    file_inst << NR_TASKLETS;
#endif

    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
#if PERF == 1
        file_cycles << ", " << cycles[dpu][0];

        double exec_time = (double) cycles[dpu][0] / clocks_sec[0][0];
        file_time << ", " << exec_time;
#else
        file_inst << ", " << cycles[dpu][0];
#endif
    }

#if PERF == 1
    file_cycles << "\n";
    file_time << "\n";
#else
    file_inst << "\n";
#endif
}