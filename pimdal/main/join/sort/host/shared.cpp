#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#include <parallel/algorithm>
#endif

#include <arrow/api.h>

#include "datatype.h"
#include "join.h"
#include "transfer_helper.h"

#ifndef NR_DPU
#define NR_DPU 4
#endif
#ifndef INNER_SIZE
#define INNER_SIZE 200000
#endif
#ifndef OUTER_SIZE
#define OUTER_SIZE 2000000
#endif
#ifndef OUTER_RANGE
#define OUTER_RANGE 1
#endif
#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

std::shared_ptr<arrow::Table> inner_table;
std::shared_ptr<arrow::Table> outer_table;

void init_buffer() {
    arrow::Result<std::unique_ptr<arrow::Buffer>> inner_try = arrow::AllocateBuffer((uint64_t) NR_DPU*INNER_SIZE*sizeof(uint32_t));
    if (!inner_try.ok()) {
        std::cout << "Could not allocate buffer!" << std::endl;
    }
    std::shared_ptr<arrow::Buffer> inner = *std::move(inner_try);

    uint32_t* inner_data = (uint32_t*) inner->mutable_data();
    uint64_t inner_size = inner->size() / sizeof(uint32_t);

    #pragma omp parallel for
    for (uint64_t i = 0; i < (uint64_t) NR_DPU*INNER_SIZE; i++) {
            inner_data[i] = i+1;
    }

    //std::default_random_engine gen(0);
    __gnu_parallel::random_shuffle(inner_data, inner_data+inner_size);

    arrow::Result<std::unique_ptr<arrow::Buffer>> outer_try = arrow::AllocateBuffer((uint64_t) NR_DPU*OUTER_SIZE*sizeof(uint32_t));
    if (!outer_try.ok()) {
        std::cout << "Could not allocate buffer!" << std::endl;
    }
    std::shared_ptr<arrow::Buffer> outer = *std::move(outer_try);

    uint32_t* outer_data = (uint32_t*) outer->mutable_data();

    #pragma omp parallel
    {
        uint32_t seed = 0;
    #ifdef _OPENMP
        seed += omp_get_thread_num();
    #endif
        std::default_random_engine gen(seed);
        std::uniform_int_distribution<uint32_t> dist(1, NR_DPU*OUTER_RANGE*INNER_SIZE);

    #pragma omp for schedule(static)
    for (uint64_t i = 0; i < (uint64_t) NR_DPU*OUTER_SIZE; i++) {
            outer_data[i] = dist(gen);
        }
    }

    // Create the inner table
    auto schema = arrow::schema({arrow::field("key", arrow::uint32(), false)});
    auto inner_key_data = arrow::ArrayData::Make(arrow::uint32(), NR_DPU*INNER_SIZE, {nullptr, inner});
    auto inner_key_array = arrow::MakeArray(inner_key_data);

    arrow::ArrayVector inner_data_vec;
    inner_data_vec.push_back(inner_key_array);
    inner_table = arrow::Table::Make(schema, inner_data_vec);

    // Create the outer table
    auto outer_key_data = arrow::ArrayData::Make(arrow::uint32(), NR_DPU*OUTER_SIZE, {nullptr, outer});
    auto outer_key_array = arrow::MakeArray(outer_key_data);

    arrow::ArrayVector outer_data_vec;
    outer_data_vec.push_back(outer_key_array);
    outer_table = arrow::Table::Make(schema, outer_data_vec);
}

void validate(std::shared_ptr<arrow::Buffer> map) {

    bool error = false;
    uint32_t* map_data = (uint32_t*) map->data();
    const uint32_t* outer_col = outer_table->GetColumnByName("key")->chunk(0)->data()->GetValues<uint32_t>(1);
    const uint32_t* inner_col = inner_table->GetColumnByName("key")->chunk(0)->data()->GetValues<uint32_t>(1);

    for (uint32_t row = 0; row < OUTER_SIZE; row++) {
        uint32_t key_outer = outer_col[row];
        uint32_t key_inner = inner_col[map_data[row]];

        if (key_outer != key_inner) {
            std::cout << "Error row " << row << ": " << key_outer << " - " << key_inner << std::endl;
            error = true;
        }
    }

    if (!error) {
        std::cout << "No error!" << std::endl;
    }
    else {
        std::cout << "Incorrect result" << std::endl;
    }
}

void output_perf_part(dpu_set_t &system) {
#if PERF == 1
    std::ofstream file_inner_part_cycles("inner_spart_cycles.csv", std::ofstream::app);
    std::ofstream file_outer_part_cycles("outer_spart_cycles.csv", std::ofstream::app);
    std::ofstream file_inner_part_time("inner_spart_time.csv", std::ofstream::app);
    std::ofstream file_outer_part_time("outer_spart_time.csv", std::ofstream::app);

    file_inner_part_cycles << NR_TASKLETS;
    file_outer_part_cycles << NR_TASKLETS;
    file_inner_part_time << NR_TASKLETS;
    file_outer_part_time << NR_TASKLETS;

    std::vector<std::vector<uint32_t>> clocks_sec(NR_DPU, std::vector<uint32_t>(1));
    get_vec(system, clocks_sec, 0, "CLOCKS_PER_SEC", DPU_XFER_DEFAULT);
#else
    std::ofstream file_inner_part_inst("inner_spart_inst.csv", std::ofstream::app);
    std::ofstream file_outer_part_inst("outer_spart_inst.csv", std::ofstream::app);

    file_inner_part_inst << NR_TASKLETS;
    file_outer_part_inst << NR_TASKLETS;
#endif

    std::vector<std::vector<uint64_t>> cycles(NR_DPU, std::vector<uint64_t>(1));
    get_vec(system, cycles, 0, "cycles_1", DPU_XFER_DEFAULT);

    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
#if PERF == 1
        file_inner_part_cycles << ", " << cycles[dpu][0];

        double exec_time = (double) cycles[dpu][0] / clocks_sec[0][0];
        file_inner_part_time << ", " << exec_time;
#else
        file_inner_part_inst << ", " << cycles[dpu][0];
#endif
    }

    get_vec(system, cycles, 0, "cycles_2", DPU_XFER_DEFAULT);

    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
#if PERF == 1
        file_outer_part_cycles << ", " << cycles[dpu][0];

        double exec_time = (double) cycles[dpu][0] / clocks_sec[0][0];
        file_outer_part_time << ", " << exec_time;
#else
        file_outer_part_inst << ", " << cycles[dpu][0];
#endif
    }

#if PERF == 1
    file_inner_part_cycles << "\n";
    file_outer_part_cycles << "\n";
    file_inner_part_time << "\n";
    file_outer_part_time << "\n";
#else
    file_inner_part_inst << "\n";
    file_outer_part_inst << "\n";
#endif
}

void output_perf_merge(dpu_set_t &system) {
#if PERF == 1
    std::ofstream file_inner_sort_cycles("inner_sort_cycles.csv", std::ofstream::app);
    std::ofstream file_outer_sort_cycles("outer_sort_cycles.csv", std::ofstream::app);
    std::ofstream file_smerge_cycles("outer_smerge_cycles.csv", std::ofstream::app);
    std::ofstream file_inner_sort_time("inner_sort_time.csv", std::ofstream::app);
    std::ofstream file_outer_sort_time("outer_sort_time.csv", std::ofstream::app);
    std::ofstream file_smerge_time("outer_smerge_time.csv", std::ofstream::app);

    file_inner_sort_cycles << NR_TASKLETS;
    file_outer_sort_cycles << NR_TASKLETS;
    file_smerge_cycles << NR_TASKLETS;
    file_inner_sort_time << NR_TASKLETS;
    file_outer_sort_time << NR_TASKLETS;
    file_smerge_time << NR_TASKLETS;

    std::vector<std::vector<uint32_t>> clocks_sec(NR_DPU, std::vector<uint32_t>(1));
    get_vec(system, clocks_sec, 0, "CLOCKS_PER_SEC", DPU_XFER_DEFAULT);
#else
    std::ofstream file_inner_sort_inst("inner_sort_inst.csv", std::ofstream::app);
    std::ofstream file_outer_sort_inst("outer_sort_inst.csv", std::ofstream::app);
    std::ofstream file_smerge_inst("outer_smerge_inst.csv", std::ofstream::app);

    file_inner_sort_inst << NR_TASKLETS;
    file_outer_sort_inst << NR_TASKLETS;
    file_smerge_inst << NR_TASKLETS;
#endif

    std::vector<std::vector<uint64_t>> cycles(NR_DPU, std::vector<uint64_t>(1));
    get_vec(system, cycles, 0, "cycles_1", DPU_XFER_DEFAULT);

    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
#if PERF == 1
        file_inner_sort_cycles << ", " << cycles[dpu][0];

        double exec_time = (double) cycles[dpu][0] / clocks_sec[0][0];
        file_inner_sort_time << ", " << exec_time;
#else
        file_inner_sort_inst << ", " << cycles[dpu][0];
#endif
    }

    get_vec(system, cycles, 0, "cycles_2", DPU_XFER_DEFAULT);

    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
#if PERF == 1
        file_outer_sort_cycles << ", " << cycles[dpu][0];

        double exec_time = (double) cycles[dpu][0] / clocks_sec[0][0];
        file_outer_sort_time << ", " << exec_time;
#else
        file_outer_sort_inst << ", " << cycles[dpu][0];
#endif
    }

    get_vec(system, cycles, 0, "cycles_3", DPU_XFER_DEFAULT);

    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
#if PERF == 1
        file_smerge_cycles << ", " << cycles[dpu][0];

        double exec_time = (double) cycles[dpu][0] / clocks_sec[0][0];
        file_smerge_time << ", " << exec_time;
#else
        file_smerge_inst << ", " << cycles[dpu][0];
#endif
    }

#if PERF == 1
    file_inner_sort_cycles << "\n";
    file_outer_sort_cycles << "\n";
    file_smerge_cycles << "\n";
    file_inner_sort_time << "\n";
    file_outer_sort_time << "\n";
    file_smerge_time << "\n";
#else
    file_inner_sort_inst << "\n";
    file_outer_sort_inst << "\n";
    file_smerge_inst << "\n";
#endif

}