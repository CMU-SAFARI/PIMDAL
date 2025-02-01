#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/acero/exec_plan.h>
#include <arrow/dataset/api.h>

#include "transfer_helper.h"
#include "datatype.h"
#include "select.h"

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

std::shared_ptr<arrow::Table> table;

void init_buffer() {

    std::shared_ptr<arrow::Buffer> buffer;

    arrow::Result<std::unique_ptr<arrow::Buffer>> buffer_try = arrow::AllocateBuffer((uint64_t) NR_DPU*BUFFER_SIZE*sizeof(int32_t));
    if (!buffer_try.ok()) {
        std::cout << "Could not allocate buffer!" << std::endl;
    }
    buffer = *std::move(buffer_try);

    int32_t* buffer_data = (int32_t*) buffer->mutable_data();
    uint64_t buffer_size = buffer->size() / sizeof(int32_t);

    #pragma omp parallel
    {
        uint32_t seed = 0;
    #ifdef _OPENMP
        seed += omp_get_thread_num();
    #endif
        std::default_random_engine gen(seed);
        std::uniform_int_distribution<int32_t> dist(1, 50);

    #pragma omp for schedule(static)
        for(uint64_t j = 0; j < buffer_size; j++) {
            buffer_data[j] = dist(gen);
        }
    }

    auto schema = arrow::schema({arrow::field("key", arrow::int32(), false)});

    auto key_data = arrow::ArrayData::Make(arrow::int32(), (uint64_t) NR_DPU*BUFFER_SIZE, {nullptr, buffer});
    auto key_array = arrow::MakeArray(key_data);

    arrow::ArrayVector data_vec;
    data_vec.push_back(key_array);

    table = arrow::Table::Make(schema, data_vec);
}

void validate(std::shared_ptr<arrow::ChunkedArray> results) {

    auto pred = arrow::compute::and_(arrow::compute::greater_equal(arrow::compute::field_ref("key"),
                                                                   arrow::compute::literal(10)),
                                     arrow::compute::less_equal(arrow::compute::field_ref("key"),
                                                                arrow::compute::literal(20)));

    ac::Declaration plan = ac::Declaration::Sequence({{"table_source", ac::TableSourceNodeOptions(table)},
                                                      {"filter", ac::FilterNodeOptions(pred)}});

    ac::QueryOptions query_options;
    query_options.use_threads = true;

    std::chrono::steady_clock::time_point begin_comp = std::chrono::steady_clock::now();
    auto actual = ac::DeclarationToTable(plan, query_options);
    auto res_table = actual.ValueOrDie();
    std::chrono::steady_clock::time_point end_comp = std::chrono::steady_clock::now();
    std::cout << "Comparison elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_comp - begin_comp).count()
              << " millisecs." << std::endl;
    
    std::cout << "\n";
    std::cout << "Length DPU result: " << results->length();
    std::cout << " Length CPU: " << res_table->GetColumnByName("key")->length() << std::endl;
    if (results->Equals(res_table->GetColumnByName("key"))) {
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
    std::ofstream file_cycles("select_cycles.csv", std::ofstream::app);
    std::ofstream file_time("select_time.csv", std::ofstream::app);

    file_cycles << NR_TASKLETS;
    file_time << NR_TASKLETS;

    std::vector<std::vector<uint32_t>> clocks_sec(NR_DPU, std::vector<uint32_t>(1));
    get_vec(system, clocks_sec, 0, "CLOCKS_PER_SEC", DPU_XFER_DEFAULT);
#else
    std::ofstream file_inst("select_inst.csv", std::ofstream::app);

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