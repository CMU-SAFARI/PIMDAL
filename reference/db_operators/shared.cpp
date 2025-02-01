#include <random>
#ifdef _OPENMP
#include <omp.h>
#include <parallel/algorithm>
#endif

#include <arrow/api.h>

template <typename T>
std::shared_ptr<arrow::Buffer> create_rand_buff(T min, T max, uint64_t size) {

    std::shared_ptr<arrow::Buffer> buffer;

    arrow::Result<std::unique_ptr<arrow::Buffer>> buffer_try = arrow::AllocateBuffer(size*sizeof(T));
    if (!buffer_try.ok()) {
        std::cout << "Could not allocate buffer!" << std::endl;
    }
    buffer = *std::move(buffer_try);

    T* buffer_data = (T*) buffer->mutable_data();

    #pragma omp parallel
    {
    #ifdef _OPENMP
        uint64_t seed = omp_get_thread_num();
    #else
        uint64_t seed = 0;
    #endif
        std::default_random_engine gen(seed);
        std::uniform_int_distribution<T> dist(min, max);

    #pragma omp for
        for(uint64_t j = 0; j < size; j++) {
            buffer_data[j] = dist(gen);
        }
    }

    return buffer;
}

template <typename T>
std::shared_ptr<arrow::Buffer> create_shuf_buf(uint64_t size) {
    arrow::Result<std::unique_ptr<arrow::Buffer>> buffer_try = arrow::AllocateBuffer(size*sizeof(T));
    if (!buffer_try.ok()) {
        std::cout << "Could not allocate buffer!" << std::endl;
    }
    std::shared_ptr<arrow::Buffer> buffer = *std::move(buffer_try);

    T* buffer_data = (T*) buffer->mutable_data();

    #pragma omp parallel for
    for (uint64_t i = 0; i < size; i++) {
            buffer_data[i] = i+1;
    }

    //std::default_random_engine gen(0);
    __gnu_parallel::random_shuffle(buffer_data, buffer_data+size);

    return buffer;
}
