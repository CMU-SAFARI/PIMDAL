#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <parallel/algorithm>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/acero/exec_plan.h>
#include <arrow/dataset/api.h>

#include "datatype.h"
#include "transfer_helper.h"
#include "args.h"

#ifndef NR_DPU
#define NR_DPU 4
#endif
#ifndef BUFFER_SIZE
#define BUFFER_SIZE 200000
#endif
#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

namespace ac = arrow::acero;

std::shared_ptr<arrow::Table> table;

void init_buffer() {

    std::shared_ptr<arrow::Buffer> buffer;

    arrow::Result<std::unique_ptr<arrow::Buffer>> buffer_try = arrow::AllocateBuffer((uint64_t) NR_DPU*BUFFER_SIZE*sizeof(uint32_t));
    if (!buffer_try.ok()) {
        std::cout << "Could not allocate buffer!" << std::endl;
    }
    buffer = *std::move(buffer_try);

    uint32_t* buffer_data = (uint32_t*) buffer->mutable_data();
    uint64_t buffer_size = buffer->size() / sizeof(uint32_t);

    #pragma omp parallel
    {
        uint32_t seed = 0;
    #ifdef _OPENMP
        seed += omp_get_thread_num();
    #endif
        std::default_random_engine gen(seed);
        std::uniform_int_distribution<uint32_t> dist(1, (uint32_t) -1);

    #pragma omp for schedule(static)
        for(uint64_t j = 0; j < buffer_size; j++) {
            buffer_data[j] = dist(gen);
        }
    }

    auto schema = arrow::schema({arrow::field("key", arrow::uint32(), false)});

    auto key_data = arrow::ArrayData::Make(arrow::uint32(), NR_DPU*BUFFER_SIZE, {nullptr, buffer});
    auto key_array = arrow::MakeArray(key_data);

    arrow::ArrayVector data_vec;
    data_vec.push_back(key_array);

    table = arrow::Table::Make(schema, data_vec);
}

void validate(std::shared_ptr<arrow::ChunkedArray> results) {
/*     auto order_by_options = ac::OrderByNodeOptions({{arrow::compute::SortKey("key", arrow::compute::SortOrder::Ascending)}});
    std::shared_ptr<arrow::Table> comp_table;
    auto plan = ac::Declaration::Sequence({{"table_source", ac::TableSourceNodeOptions(table)},
                                           {"order_by_sink", order_by_options}});

    ac::QueryOptions query_options;

    query_options.use_threads = true; */

    // Sorting workaround since the apache arrow function is not multithreaded
    std::shared_ptr<arrow::ArrayData> data = table->GetColumnByName("key")->chunk(0)->data();
    uint32_t* begin = data->GetMutableValues<uint32_t>(1);
    uint32_t* end = begin + table->GetColumnByName("key")->length();

    std::chrono::steady_clock::time_point begin_comp = std::chrono::steady_clock::now();
    #ifdef _OPENMP
    __gnu_parallel::sort(begin, end);
    #else
    std::sort(begin, end);
    #endif
    std::chrono::steady_clock::time_point end_comp = std::chrono::steady_clock::now();

    std::cout << "Comparison elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_comp - begin_comp).count()
              << " millisecs." << std::endl;
    
    if (results->Equals(table->GetColumnByName("key"))) {
        std::cout << "No error!" << std::endl;
    }
    else {
        std::cout << "Incorrect result" << std::endl;
    }
}

void output_perf(dpu_set_t &system, bool part) {
    std::vector<std::vector<uint64_t>> cycles(NR_DPU, std::vector<uint64_t>(1));
    get_vec(system, cycles, 0, "cycles", DPU_XFER_DEFAULT);

#if PERF == 1
    std::ofstream file_cycles;
    std::ofstream file_time;

    if (part) {
        file_cycles.open("sort_part_cycles.csv", std::ofstream::app);
        file_time.open("sort_part_time.csv", std::ofstream::app);
    }
    else {
        file_cycles.open("sort_cycles.csv", std::ofstream::app);
        file_time.open("sort_time.csv", std::ofstream::app);
    }

    file_cycles << NR_TASKLETS;
    file_time << NR_TASKLETS;

    std::vector<std::vector<uint32_t>> clocks_sec(NR_DPU, std::vector<uint32_t>(1));
    get_vec(system, clocks_sec, 0, "CLOCKS_PER_SEC", DPU_XFER_DEFAULT);
#else
    std::ofstream file_inst;

    if (part) {
        file_inst.open("sort_part_inst.csv", std::ofstream::app);
    }
    else {
        file_inst.open("sort_inst.csv", std::ofstream::app);
    }

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