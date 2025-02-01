#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "shared.cpp"

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 400000
#endif

#define BLOCK_SIZE 8000
#define NUM_BLOCKS (BUFFER_SIZE/BLOCK_SIZE)

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
void sort_merge_join_kernel(T* inner_in, T* outer_in, T* inner_res, T* outer_res,
                            uint64_t inner_size, uint64_t outer_size) {

    std::shared_ptr<arrow::Buffer> inner_sort_buf;

    arrow::Result<std::unique_ptr<arrow::Buffer>> inner_sort_buf_try = arrow::AllocateBuffer(inner_size*sizeof(uint32_t));
    if (!inner_sort_buf_try.ok()) {
        std::cout << "Could not allocate buffer!" << std::endl;
    }
    inner_sort_buf = *std::move(inner_sort_buf_try);

    uint32_t* inner_data = (uint32_t*) inner_sort_buf->mutable_data();

    std::copy(inner_in, inner_in+inner_size, inner_data);

    // Sort inner table
    #ifdef _OPENMP
    __gnu_parallel::sort(inner_data, inner_data+inner_size);
    #else
    std::sort(inner_data, inner_data+inner_size);
    #endif

    std::copy(outer_in, outer_in+outer_size, outer_res);

    // Sort outer table
    #ifdef _OPENMP
    __gnu_parallel::sort(outer_res, outer_res+outer_size);
    #else
    std::sort(outer_res, outer_res+outer_size);
    #endif

    // Merge tables
    T* inner_it = inner_data;
    T* inner_end = inner_data + inner_size;
    T* outer_res_end = outer_res + outer_size;
    #pragma omp parallel for
    for (uint64_t block = 0; block < NUM_BLOCKS; block++) {
        T* block_begin = outer_res + block*BLOCK_SIZE;
        T* block_end = outer_res + BLOCK_SIZE > outer_res_end ? outer_res_end : block_begin + BLOCK_SIZE;

        T* res_it = inner_res + block*BLOCK_SIZE;

        for (T* outer_it = block_begin; outer_it < block_end; outer_it++) {
            // Iterate through inner buffer
            while (*outer_it > *inner_it && inner_it < inner_end) {
                inner_it++;
            }

            // Match in outer and inner buffer
            bool hit = (*outer_it == *inner_it);
            *res_it = *inner_it;
            res_it += hit;
        }
    }
}

std::shared_ptr<arrow::Table> sort_merge_join(std::shared_ptr<arrow::Table> inner_table,
                                              std::shared_ptr<arrow::Table> outer_table,
                                              std::string column_name) {

    std::shared_ptr<arrow::Array> inner_column = inner_table->GetColumnByName(column_name)->chunk(0);
    std::shared_ptr<arrow::Array> outer_column = outer_table->GetColumnByName(column_name)->chunk(0);

    uint32_t* inner_begin = inner_column->data()->GetMutableValues<uint32_t>(1);
    uint64_t inner_size = outer_table->GetColumnByName(column_name)->length();

    uint32_t* outer_begin = outer_column->data()->GetMutableValues<uint32_t>(1);
    uint64_t outer_size = outer_table->GetColumnByName(column_name)->length();

    std::shared_ptr<arrow::Buffer> inner_res_buf;

    arrow::Result<std::unique_ptr<arrow::Buffer>> inner_res_buf_try = arrow::AllocateBuffer(inner_size*sizeof(uint32_t));
    if (!inner_res_buf_try.ok()) {
        std::cout << "Could not allocate buffer!" << std::endl;
    }
    inner_res_buf = *std::move(inner_res_buf_try);

    uint32_t* inner_res_buf_data = (uint32_t*) inner_res_buf->mutable_data();

    std::shared_ptr<arrow::Buffer> outer_res_buf;

    arrow::Result<std::unique_ptr<arrow::Buffer>> outer_res_buf_try = arrow::AllocateBuffer(outer_size*sizeof(uint32_t));
    if (!outer_res_buf_try.ok()) {
        std::cout << "Could not allocate buffer!" << std::endl;
    }
    outer_res_buf = *std::move(outer_res_buf_try);

    uint32_t* outer_res_buf_data = (uint32_t*) outer_res_buf->mutable_data();

    sort_merge_join_kernel(inner_begin, outer_begin, inner_res_buf_data, outer_res_buf_data, inner_size, outer_size);

    arrow::ArrayVector data_vec;

    auto inner_res_data = arrow::ArrayData::Make(arrow::uint32(), inner_size, {nullptr, inner_res_buf});
    auto inner_res_array = arrow::MakeArray(inner_res_data);
    
    auto outer_res_data = arrow::ArrayData::Make(arrow::uint32(), outer_size, {nullptr, outer_res_buf});
    auto outer_res_array = arrow::MakeArray(outer_res_data);

    data_vec.push_back(inner_res_array);
    data_vec.push_back(outer_res_array);

    auto schema = arrow::schema({arrow::field("inner_key", arrow::uint32(), false),
                                 arrow::field("outer_key", arrow::uint32(), false)});
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
        result = sort_merge_join(inner_table, outer_table, "key");
    }
    std::chrono::steady_clock::time_point end_join = std::chrono::steady_clock::now();


    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_join - begin_join);
    double join_time = duration.count() / repetitions;
    std::cout << "Sort-merge join time: "
            << join_time
            << " millisecs." << std::endl;

    //std::cout << result->ToString() << std::endl;
}