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

std::shared_ptr<arrow::Table> create_table() {
    std::shared_ptr<arrow::Buffer> buffer = create_rand_buff<uint32_t>(0, 0xffffffff, BUFFER_SIZE);
    auto schema = arrow::schema({arrow::field("key", arrow::uint32(), false)});

    auto key_data = arrow::ArrayData::Make(arrow::uint32(), BUFFER_SIZE, {nullptr, buffer});
    auto key_array = arrow::MakeArray(key_data);

    arrow::ArrayVector data_vec;
    data_vec.push_back(key_array);

    std::shared_ptr<arrow::Table> inner_table = arrow::Table::Make(schema, data_vec);

    return inner_table;
}

template <typename T>
void sort_kernel(T* in, T* out, uint64_t size) {

    std::copy(in, in+size, out);

    #ifdef _OPENMP
    __gnu_parallel::sort(out, out+size);
    #else
    std::sort(out, out+size);
    #endif
}

std::shared_ptr<arrow::Table> sort(std::shared_ptr<arrow::Table> table, std::string column_name) {
    std::shared_ptr<arrow::Array> column = table->GetColumnByName(column_name)->chunk(0);

    uint32_t* in = column->data()->GetMutableValues<uint32_t>(1);
    uint64_t size = + table->GetColumnByName(column_name)->length();

    std::shared_ptr<arrow::Buffer> res_buf;

    arrow::Result<std::unique_ptr<arrow::Buffer>> res_buf_try = arrow::AllocateBuffer(size*sizeof(uint32_t));
    if (!res_buf_try.ok()) {
        std::cout << "Could not allocate buffer!" << std::endl;
    }
    res_buf = *std::move(res_buf_try);

    uint32_t* res_buf_data = (uint32_t*) res_buf->mutable_data();

    sort_kernel(in, res_buf_data, size);

    auto res_data = arrow::ArrayData::Make(arrow::uint32(), size, {nullptr, res_buf});
    auto res_array = arrow::MakeArray(res_data);

    arrow::ArrayVector data_vec;
    data_vec.push_back(res_array);

    auto schema = arrow::schema({arrow::field("key", arrow::uint32(), false)});
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
    std::chrono::steady_clock::time_point begin_sort = std::chrono::steady_clock::now();
    for (uint32_t rep = 0; rep < repetitions; rep++) {
        result = sort(table, "key");
    }
    std::chrono::steady_clock::time_point end_sort = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_sort - begin_sort);
    double sort_time = duration.count() / repetitions;
    std::cout << "Sort time: "
            << sort_time
            << " millisecs." << std::endl;

    //std::cout << result->ToString() << std::endl;
}