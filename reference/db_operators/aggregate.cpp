#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include "absl/container/flat_hash_map.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#include "shared.cpp"

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 400000
#endif

#define BLOCK_SIZE 8000
#define NUM_BLOCKS (BUFFER_SIZE/BLOCK_SIZE)

std::shared_ptr<arrow::Table> create_table() {
    auto schema = arrow::schema({{arrow::field("key", arrow::uint32(), false)},
                                 {arrow::field("val", arrow::int32(), false)}});

    std::shared_ptr<arrow::Buffer> key_buffer = create_rand_buff<uint32_t>(1, 50, BUFFER_SIZE);
    auto key_data = arrow::ArrayData::Make(arrow::uint32(), BUFFER_SIZE, {nullptr, key_buffer});
    auto key_array = arrow::MakeArray(key_data);

    std::shared_ptr<arrow::Buffer> val_buffer = create_rand_buff<int32_t>(1, 10, BUFFER_SIZE);
    auto val_data = arrow::ArrayData::Make(arrow::int32(), BUFFER_SIZE, {nullptr, val_buffer});
    auto val_array = arrow::MakeArray(val_data);

    arrow::ArrayVector data_vec;
    data_vec.push_back(key_array);
    data_vec.push_back(val_array);

    std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, data_vec);

    return table;
}

template <typename T, typename U>
absl::flat_hash_map<T, U> hash_table_aggregate(const T* key_begin, const T* key_end, const U* val_begin, const U* val_end) {

    absl::flat_hash_map<T, U> hash_table;

    #pragma omp parallel
    {
        absl::flat_hash_map<T, U> hash_table_loc;

        #pragma omp for
        for (uint64_t block = 0; block < NUM_BLOCKS; block++) {
            const T* block_begin = key_begin + block*BLOCK_SIZE;
            const T* block_end = block_begin + BLOCK_SIZE > key_end ? key_end : block_begin + BLOCK_SIZE;

            const U* val_block_it = val_begin + block*BLOCK_SIZE;

            for (const T* it = block_begin; it < block_end; it++) {
                auto existing = hash_table_loc.find(*it);
                if (existing != hash_table_loc.end()) {
                    existing->second += *val_block_it;
                }
                else {
                    hash_table_loc[*it] = *val_block_it;
                }
                val_block_it++;
            }
        }

        #pragma omp critical
        {
            for (auto it = hash_table_loc.begin(); it != hash_table_loc.end(); ++it) {
                auto existing = hash_table.find(it->first);
                if (existing != hash_table.end()) {
                    existing->second += it->second;
                }
                else {
                    hash_table[it->first] = it->second;
                }
            }
        }
    }

    return hash_table;
}

template <typename T, typename U>
void hash_table_gather(absl::flat_hash_map<T, U> hash_table, T* key_it, U* val_it) {
    for (auto it = hash_table.begin(); it != hash_table.end(); ++it) {
        *key_it = it->first;
        *val_it = it->second;
        key_it++;
        val_it++;
    }
}

std::shared_ptr<arrow::Table> hash_aggr(std::shared_ptr<arrow::Table> table,
                                        std::string column_name) {
    std::shared_ptr<arrow::Array> key_col = table->GetColumnByName("key")->chunk(0);

    std::shared_ptr<arrow::ArrayData> key_data = key_col->data();
    const uint32_t* key_begin = key_data->GetValues<uint32_t>(1);
    uint64_t key_size = table->GetColumnByName("key")->length();
    const uint32_t* key_end = key_begin + key_size;

    std::shared_ptr<arrow::Array> val_col = table->GetColumnByName(column_name)->chunk(0);
    std::shared_ptr<arrow::ArrayData> val_data = val_col->data();
    const int32_t* val_begin = val_data->GetValues<int32_t>(1);
    uint64_t val_size = table->GetColumnByName(column_name)->length();
    const int32_t* val_end = val_begin + val_size;

    absl::flat_hash_map<uint32_t, int32_t> hash_table = hash_table_aggregate(key_begin, key_end, val_begin, val_end);
    uint64_t out_size = hash_table.size();

    std::shared_ptr<arrow::Buffer> key_out_buf;

    arrow::Result<std::unique_ptr<arrow::Buffer>> key_out_buf_try = arrow::AllocateBuffer(out_size*sizeof(uint32_t));
    if (!key_out_buf_try.ok()) {
        std::cout << "Could not allocate buffer!" << std::endl;
    }
    key_out_buf = *std::move(key_out_buf_try);

    uint32_t* key_buf_data = (uint32_t*) key_out_buf->mutable_data();

    std::shared_ptr<arrow::Buffer> val_out_buf;

    arrow::Result<std::unique_ptr<arrow::Buffer>> val_out_buf_try = arrow::AllocateBuffer(out_size*sizeof(uint32_t));
    if (!val_out_buf_try.ok()) {
        std::cout << "Could not allocate buffer!" << std::endl;
    }
    val_out_buf = *std::move(val_out_buf_try);

    int32_t* val_buf_data = (int32_t*) val_out_buf->mutable_data();

    hash_table_gather(hash_table, key_buf_data, val_buf_data);

    arrow::ArrayVector data_vec;
    
    auto key_out_data = arrow::ArrayData::Make(arrow::uint32(), out_size, {nullptr, key_out_buf});
    auto key_out_array = arrow::MakeArray(key_out_data);

    auto val_out_data = arrow::ArrayData::Make(arrow::int32(), out_size, {nullptr, val_out_buf});
    auto val_out_array = arrow::MakeArray(val_out_data);

    data_vec.push_back(key_out_array);
    data_vec.push_back(val_out_array);

    auto schema = arrow::schema({{arrow::field("key", arrow::uint32(), false)},
                                 {arrow::field("val", arrow::int32(), false)}});
    std::shared_ptr<arrow::Table> result = arrow::Table::Make(schema, data_vec);

    return result;
}

int main() {
    std::chrono::steady_clock::time_point begin_create = std::chrono::steady_clock::now();
    std::shared_ptr<arrow::Table> table = create_table();
    std::chrono::steady_clock::time_point end_create = std::chrono::steady_clock::now();

    std::cout << "Data creation time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_create - begin_create).count()
            << " millisecs." << std::endl;

    uint32_t repetitions = 5;
    std::shared_ptr<arrow::Table> result;
    std::chrono::steady_clock::time_point begin_aggr = std::chrono::steady_clock::now();
    for (uint32_t rep = 0; rep < repetitions; rep++) {
        result = hash_aggr(table, "val");
    }
    std::chrono::steady_clock::time_point end_aggr = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_aggr - begin_aggr);
    double aggr_time = duration.count() / repetitions;
    std::cout << "Aggregation time: "
            << aggr_time
            << " millisecs." << std::endl;

    //std::cout << result->ToString() << std::endl;
}