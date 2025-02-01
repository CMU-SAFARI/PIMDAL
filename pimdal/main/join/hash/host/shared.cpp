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

static constexpr uint32_t NR_BUCKETS = NR_DPU;
static constexpr uint32_t TABLE_SIZE = 4096*256;

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
    std::ofstream file_inner_part_cycles("inner_hpart_cycles.csv", std::ofstream::app);
    std::ofstream file_outer_part_cycles("outer_hpart_cycles.csv", std::ofstream::app);
    std::ofstream file_inner_part_time("inner_hpart_time.csv", std::ofstream::app);
    std::ofstream file_outer_part_time("outer_hpart_time.csv", std::ofstream::app);

    file_inner_part_cycles << NR_TASKLETS;
    file_outer_part_cycles << NR_TASKLETS;
    file_inner_part_time << NR_TASKLETS;
    file_outer_part_time << NR_TASKLETS;

    std::vector<std::vector<uint32_t>> clocks_sec(NR_DPU, std::vector<uint32_t>(1));
    get_vec(system, clocks_sec, 0, "CLOCKS_PER_SEC", DPU_XFER_DEFAULT);
#else
    std::ofstream file_inner_part_inst("inner_hpart_inst.csv", std::ofstream::app);
    std::ofstream file_outer_part_inst("outer_hpart_inst.csv", std::ofstream::app);

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
    std::ofstream file_hash_cycles("hash_cycles.csv", std::ofstream::app);
    std::ofstream file_hpart_cycles("hpart_cycles.csv", std::ofstream::app);
    std::ofstream file_hmerge_cycles("hmerge_cycles.csv", std::ofstream::app);
    std::ofstream file_hash_time("hash_time.csv", std::ofstream::app);
    std::ofstream file_hpart_time("hpart_time.csv", std::ofstream::app);
    std::ofstream file_hmerge_time("hmerge_time.csv", std::ofstream::app);

    file_hash_cycles << NR_TASKLETS;
    file_hpart_cycles << NR_TASKLETS;
    file_hmerge_cycles << NR_TASKLETS;
    file_hash_time << NR_TASKLETS;
    file_hpart_time << NR_TASKLETS;
    file_hmerge_time << NR_TASKLETS;

    std::vector<std::vector<uint32_t>> clocks_sec(NR_DPU, std::vector<uint32_t>(1));
    get_vec(system, clocks_sec, 0, "CLOCKS_PER_SEC", DPU_XFER_DEFAULT);
#else
    std::ofstream file_hash_inst("hash_inst.csv", std::ofstream::app);
    std::ofstream file_hpart_inst("hpart_inst.csv", std::ofstream::app);
    std::ofstream file_hmerge_inst("hmerge_inst.csv", std::ofstream::app);

    file_hash_inst << NR_TASKLETS;
    file_hpart_inst << NR_TASKLETS;
    file_hmerge_inst << NR_TASKLETS;
#endif

    std::vector<std::vector<uint64_t>> cycles(NR_DPU, std::vector<uint64_t>(1));
    get_vec(system, cycles, 0, "cycles_1", DPU_XFER_DEFAULT);

    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
#if PERF == 1
        file_hash_cycles << ", " << cycles[dpu][0];

        double exec_time = (double) cycles[dpu][0] / clocks_sec[0][0];
        file_hash_time << ", " << exec_time;
#else
        file_hash_inst << ", " << cycles[dpu][0];
#endif
    }

    get_vec(system, cycles, 0, "cycles_2", DPU_XFER_DEFAULT);

    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
#if PERF == 1
        file_hpart_cycles << ", " << cycles[dpu][0];

        double exec_time = (double) cycles[dpu][0] / clocks_sec[0][0];
        file_hpart_time << ", " << exec_time;
#else
        file_hpart_inst << ", " << cycles[dpu][0];
#endif
    }

    get_vec(system, cycles, 0, "cycles_3", DPU_XFER_DEFAULT);

    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
#if PERF == 1
        file_hmerge_cycles << ", " << cycles[dpu][0];

        double exec_time = (double) cycles[dpu][0] / clocks_sec[0][0];
        file_hmerge_time << ", " << exec_time;
#else
        file_hmerge_inst << ", " << cycles[dpu][0];
#endif
    }

#if PERF == 1
    file_hash_cycles << "\n";
    file_hpart_cycles << "\n";
    file_hmerge_cycles << "\n";
    file_hash_time << "\n";
    file_hpart_time << "\n";
    file_hmerge_time << "\n";
#else
    file_hash_inst << "\n";
    file_hpart_inst << "\n";
    file_hmerge_inst << "\n";
#endif

}