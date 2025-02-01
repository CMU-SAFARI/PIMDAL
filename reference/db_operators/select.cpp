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

std::shared_ptr<arrow::Table> create_table() {
    std::shared_ptr<arrow::Buffer> buffer = create_rand_buff<int32_t>(1, 50, BUFFER_SIZE);
    auto schema = arrow::schema({arrow::field("key", arrow::int32(), false)});

    auto key_data = arrow::ArrayData::Make(arrow::int32(), BUFFER_SIZE, {nullptr, buffer});
    auto key_array = arrow::MakeArray(key_data);

    arrow::ArrayVector data_vec;
    data_vec.push_back(key_array);

    std::shared_ptr<arrow::Table> inner_table = arrow::Table::Make(schema, data_vec);

    return inner_table;
}

template <typename T>
bool inline predicate(T element) {
    return 10 <= element && element < 30;
}

template <typename T>
uint64_t sel_cnt_kernel(const T* begin, const T* end, uint64_t counts[NUM_BLOCKS+1]) {
    uint64_t out_i = 0;

    #pragma omp parallel for reduction(+:out_i)
    for (uint64_t block = 0; block < NUM_BLOCKS; block++){
        const T* end_block = begin + (block+1)*BLOCK_SIZE;
        end_block = end_block > end ? end : end_block;
        uint64_t block_cnt = 0;

        for (const T* it = begin + block*BLOCK_SIZE; it < end_block; it++) {
            bool sel = predicate(*it);
            block_cnt += sel;
        }

        counts[block+1] = block_cnt;
        out_i += block_cnt;
    }

    return out_i;
}

template <typename T>
void sel_out_kernel(const T* begin, const T* end, T* out, uint64_t counts[NUM_BLOCKS+1]) {

    #pragma omp parallel for
    for (uint64_t block = 0; block < NUM_BLOCKS; block++){
        const T* end_block = begin + (block+1)*BLOCK_SIZE;
        end_block = end_block > end ? end : end_block;
        uint64_t block_cnt = 0;

        for (const T* it = begin + block*BLOCK_SIZE; it < end_block; it++) {
            bool sel = predicate(*it);
            out[counts[block] + block_cnt] = *it;
            block_cnt += sel;
        }
    }
}

std::shared_ptr<arrow::Table> select(std::shared_ptr<arrow::Table> table, std::string column_name) {
    std::shared_ptr<arrow::Array> column = table->GetColumnByName(column_name)->chunk(0);

    const int32_t* begin = column->data()->GetValues<int32_t>(1);
    const int32_t* end = begin + table->GetColumnByName(column_name)->length();

    uint64_t counts[NUM_BLOCKS+1];
    uint64_t selected = sel_cnt_kernel(begin, end, counts);

    counts[0] = 0;
    for (uint32_t i = 0; i < NUM_BLOCKS; i++) {
        counts[i+1] += counts[i];
    }

    std::shared_ptr<arrow::Buffer> out_buf;
    arrow::Result<std::unique_ptr<arrow::Buffer>> out_buf_try = arrow::AllocateBuffer(selected*sizeof(int32_t));
    if (!out_buf_try.ok()) {
        std::cout << "Could not allocate buffer!" << std::endl;
    }
    out_buf = *std::move(out_buf_try);

    int32_t* out_buf_data = (int32_t*) out_buf->mutable_data();

    sel_out_kernel(begin, end, out_buf_data, counts);

    auto out_data = arrow::ArrayData::Make(arrow::int32(), selected, {nullptr, out_buf});
    auto out_array = arrow::MakeArray(out_data);

    arrow::ArrayVector data_vec;
    data_vec.push_back(out_array);

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
    std::chrono::steady_clock::time_point begin_select = std::chrono::steady_clock::now();
    for (uint32_t rep = 0; rep < repetitions; rep++) {
        result = select(table, "key");
    }
    std::chrono::steady_clock::time_point end_select = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_select - begin_select);
    double select_time = duration.count() / repetitions;
    std::cout << "Selection time: "
            << select_time
            << " millisecs." << std::endl;

    //std::cout << result->ToString() << std::endl;
}