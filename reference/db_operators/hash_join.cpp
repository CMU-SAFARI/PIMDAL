#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include <absl/container/flat_hash_set.h>
#include "absl/hash/hash.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#include "shared.cpp"

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 400000
#endif

#define BLOCK_SIZE 8000
#define NUM_BLOCKS (BUFFER_SIZE/BLOCK_SIZE)

inline uint32_t hash(uint32_t input) {
    uint64_t x = static_cast<uint64_t>(input);
    constexpr uint64_t multiplier = 11400714785074694791ULL;
    uint32_t hash = static_cast<uint32_t>(__builtin_bswap64(x * multiplier));
    return hash;
}

std::shared_ptr<arrow::Table> create_inner() {
    std::shared_ptr<arrow::Buffer> inner_buffer = create_shuf_buf<uint32_t>(BUFFER_SIZE);
    auto schema = arrow::schema({arrow::field("key", arrow::uint32(), false)});

    auto key_data = arrow::ArrayData::Make(arrow::uint32(), BUFFER_SIZE, {nullptr, inner_buffer});
    auto key_array = arrow::MakeArray(key_data);

    arrow::ArrayVector data_vec;
    data_vec.push_back(key_array);

    std::shared_ptr<arrow::Table> inner_table = arrow::Table::Make(schema, data_vec);

    return inner_table;
}

std::shared_ptr<arrow::Table> create_outer() {
    std::shared_ptr<arrow::Buffer> outer_buffer = create_rand_buff<uint32_t>(1, BUFFER_SIZE, BUFFER_SIZE);
    auto schema = arrow::schema({arrow::field("key", arrow::uint32(), false)});

    auto key_data = arrow::ArrayData::Make(arrow::uint32(), BUFFER_SIZE, {nullptr, outer_buffer});
    auto key_array = arrow::MakeArray(key_data);

    arrow::ArrayVector data_vec;
    data_vec.push_back(key_array);

    std::shared_ptr<arrow::Table> outer_table = arrow::Table::Make(schema, data_vec);

    return outer_table;
}

template <typename T>
arrow::BufferVector partition(const T* begin, const T* end, uint32_t part_n) {
    std::vector<uint64_t> part_index(part_n * 64, 0);
    arrow::BufferVector partitions(part_n);

    #pragma omp parallel
    {
        uint32_t thread_id = omp_get_thread_num();
        uint32_t thread_n = omp_get_num_threads();

        #pragma omp for
        for (uint64_t block = 0; block < NUM_BLOCKS; block++){
            const T* end_block = begin + (block+1)*BLOCK_SIZE;
            end_block = end_block > end ? end : end_block;
            uint64_t block_cnt = 0;

            for (const T* it = begin + block*BLOCK_SIZE; it < end_block; it++) {
                uint32_t bucket = hash(*it) % part_n;
                part_index[bucket * (thread_n+1) + thread_id+1]++;
            }
        }

        #pragma omp for
        for (uint32_t bucket = 0; bucket < part_n; bucket++) {
            for (uint32_t i = 0; i < thread_n; i++) {
                part_index[bucket * (thread_n+1) + i+1] += part_index[bucket * (thread_n+1) + i];
            }
        }

        #pragma omp for
        for (uint32_t bucket = 0; bucket < part_n; bucket++) {

            uint64_t size = part_index[(bucket+1) * (thread_n+1) - 1];
            arrow::Result<std::unique_ptr<arrow::Buffer>> buffer_try = arrow::AllocateBuffer(size*sizeof(T));
            if (!buffer_try.ok()) {
                std::cout << "Could not allocate buffer!" << std::endl;
            }

            partitions[bucket] = *std::move(buffer_try);
        }

        std::vector<T*> data_out(part_n);
        for (uint32_t bucket = 0; bucket < part_n; bucket++) {
            uint64_t offset = part_index[bucket * (thread_n+1) + thread_id];
            data_out[bucket] = ((T*) partitions[bucket]->mutable_data()) + offset;
        }

        #pragma omp for
        for (uint64_t block = 0; block < NUM_BLOCKS; block++){
            const T* end_block = begin + (block+1)*BLOCK_SIZE;
            end_block = end_block > end ? end : end_block;
            uint64_t block_cnt = 0;

            for (const T* it = begin + block*BLOCK_SIZE; it < end_block; it++) {
                uint64_t bucket = hash(*it) % part_n;
                *data_out[bucket] = *it;
                data_out[bucket]++;
            }
        }
    }

    return partitions;
}

template <typename T>
uint64_t join_partitions(arrow::BufferVector inner_part, arrow::BufferVector outer_part, uint32_t part_n,
                         T* inner_res_it, T* outer_res_it) {

    uint64_t joined_size = 0;

    #pragma omp parallel for reduction(+: joined_size)
    for (uint32_t bucket = 0; bucket < part_n; bucket++) {
        uint64_t inner_size = inner_part[bucket]->size() / sizeof(T);
        T* inner_data = (T*) inner_part[bucket]->mutable_data();
        absl::flat_hash_set<T> table;

        for (uint64_t i = 0; i < inner_size; i++) {
            table.insert(inner_data[i]);
        }

        uint64_t offset = 0;
        for (uint64_t j = 0; j < bucket; j++) {
            offset += outer_part[j]->size() / sizeof(T);
        }

        uint64_t outer_size = outer_part[bucket]->size() / sizeof(T);
        T* outer_data = (T*) outer_part[bucket]->mutable_data();

        for (uint64_t i = 0; i < outer_size; i++) {
            auto existing = table.find(outer_data[i]);
            if (existing != table.end()) {
                inner_res_it[offset] = *existing;
                outer_res_it[offset] = outer_data[i];
                offset++;
                joined_size++;
            }
        }
    }

    return joined_size;
}

std::shared_ptr<arrow::Table> hash_join(std::shared_ptr<arrow::Table> inner_table,
                                              std::shared_ptr<arrow::Table> outer_table,
                                              std::string column_name) {

    uint64_t part_n = 16384;

    std::shared_ptr<arrow::ArrayData> inner_data = inner_table->GetColumnByName(column_name)->chunk(0)->data();
    const uint32_t* inner_begin = inner_data->GetValues<uint32_t>(1);
    uint64_t inner_size = inner_table->GetColumnByName(column_name)->length();
    const uint32_t* inner_end = inner_begin + inner_size;

    arrow::BufferVector inner_part = partition(inner_begin, inner_end, part_n);

    std::shared_ptr<arrow::ArrayData> outer_data = outer_table->GetColumnByName(column_name)->chunk(0)->data();
    const uint32_t* outer_begin = outer_data->GetValues<uint32_t>(1);
    uint64_t outer_size = outer_table->GetColumnByName(column_name)->length();
    const uint32_t* outer_end = outer_begin + outer_size;

    arrow::BufferVector outer_part = partition(outer_begin, outer_end, part_n);

    // Create the buffer for the outer relation result
    std::shared_ptr<arrow::Buffer> inner_res_buf;

    arrow::Result<std::unique_ptr<arrow::Buffer>> inner_res_buf_try = arrow::AllocateBuffer(outer_size*sizeof(uint32_t));
    if (!inner_res_buf_try.ok()) {
        std::cout << "Could not allocate buffer!" << std::endl;
    }
    inner_res_buf = *std::move(inner_res_buf_try);

    uint32_t* inner_res_it = (uint32_t*) inner_res_buf->mutable_data();

    // Create the buffer for the outer relation result
    std::shared_ptr<arrow::Buffer> outer_res_buf;

    arrow::Result<std::unique_ptr<arrow::Buffer>> outer_res_buf_try = arrow::AllocateBuffer(outer_size*sizeof(uint32_t));
    if (!outer_res_buf_try.ok()) {
        std::cout << "Could not allocate buffer!" << std::endl;
    }
    outer_res_buf = *std::move(outer_res_buf_try);

    uint32_t* outer_res_it = (uint32_t*) outer_res_buf->mutable_data();

    uint64_t joined_size = join_partitions(inner_part, outer_part, part_n, inner_res_it, outer_res_it);

    arrow::ArrayVector data_vec;
    
    auto inner_res_data = arrow::ArrayData::Make(arrow::uint32(), joined_size, {nullptr, inner_res_buf});
    auto inner_res_array = arrow::MakeArray(inner_res_data);

    auto outer_res_data = arrow::ArrayData::Make(arrow::uint32(), joined_size, {nullptr, outer_res_buf});
    auto outer_res_array = arrow::MakeArray(outer_res_data);

    data_vec.push_back(inner_res_array);
    data_vec.push_back(outer_res_array);

    auto schema = arrow::schema({{arrow::field("inner_key", arrow::uint32(), false)},
                                 {arrow::field("outer_key", arrow::uint32(), false)}});
    std::shared_ptr<arrow::Table> result = arrow::Table::Make(schema, data_vec);

    return result;
}

int main() {
    std::chrono::steady_clock::time_point begin_create = std::chrono::steady_clock::now();
    std::shared_ptr<arrow::Table> inner_table = create_inner();
    std::shared_ptr<arrow::Table> outer_table = create_outer();
    std::chrono::steady_clock::time_point end_create = std::chrono::steady_clock::now();

    std::cout << "Data creation time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_create - begin_create).count()
            << " millisecs." << std::endl;

    uint32_t repetitions = 5;
    std::shared_ptr<arrow::Table> result;
    std::chrono::steady_clock::time_point begin_join = std::chrono::steady_clock::now();
    for (uint32_t rep = 0; rep < repetitions; rep++) {
        result = hash_join(inner_table, outer_table, "key");
    }
    std::chrono::steady_clock::time_point end_join = std::chrono::steady_clock::now();


    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_join - begin_join);
    double join_time = duration.count() / repetitions;
    std::cout << "Hash join time: "
            << join_time
            << " millisecs." << std::endl;

    //std::cout << result->ToString() << std::endl;
}