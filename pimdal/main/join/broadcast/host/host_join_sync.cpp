#include "shared.cpp"

void populate_mram(dpu_set_t &system) {

    std::vector<std::vector<join_arguments_t>> join_args(NR_DPU, std::vector<join_arguments_t>(1));
    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        join_args[dpu][0].kernel_sel = 0;
        join_args[dpu][0].ptr_inner = dpu*INNER_SIZE;
        join_args[dpu][0].ptr_outer = dpu*OUTER_SIZE;
    }
    dist_vec(system, join_args, 0, "join_args", DPU_XFER_DEFAULT);

    dist_table<uint32_t>(system, inner_table, "key", DPU_MRAM_HEAP_POINTER_NAME, 0, DPU_XFER_DEFAULT);

    dist_table<uint32_t>(system, outer_table, "key", DPU_MRAM_HEAP_POINTER_NAME, OUTER_SIZE*sizeof(key_ptr32), DPU_XFER_DEFAULT);
}

std::vector<std::vector<join_arguments_t>> redistribute(dpu_set_t &system, uint32_t inner_off) {

    std::vector<std::vector<join_arguments_t>> join_args (NR_DPU, std::vector<join_arguments_t>(1));

    arrow::Result<std::unique_ptr<arrow::Buffer>> inner_try = arrow::AllocateBuffer(NR_DPU*INNER_SIZE*sizeof(key_ptr32));
        if (!inner_try.ok()) {
            std::cout << "Could not allocate buffer!" << std::endl;
        }
    std::shared_ptr<arrow::Buffer> inner = *std::move(inner_try);

    std::vector<uint64_t> offset (NR_DPU+1, 0);
    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        offset[dpu + 1] = offset[dpu] + INNER_SIZE*sizeof(key_ptr32);

        join_args[dpu][0].kernel_sel = 1;
        join_args[dpu][0].n_el_inner = NR_DPU*INNER_SIZE;
        join_args[dpu][0].n_el_outer = OUTER_SIZE;
        join_args[dpu][0].range = NR_DPU*INNER_SIZE;
        join_args[dpu][0].start = 0;
    }

    collect_buf(system, inner, inner_off, DPU_MRAM_HEAP_POINTER_NAME, INNER_SIZE*sizeof(key_ptr32), offset, DPU_SG_XFER_DEFAULT);

    copy_buf(system, inner, OUTER_SIZE*sizeof(key_ptr32), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    dist_vec(system, join_args, 0, "join_args", DPU_XFER_DEFAULT);

    return join_args;
}

std::shared_ptr<arrow::Buffer> get_results(dpu_set_t &system) {
    std::vector<std::vector<join_results_t>> join_res (NR_DPU, std::vector<join_results_t>(1));
    arrow::ArrayVector results_chunks;

    get_vec(system, join_res, 0, "join_res", DPU_XFER_DEFAULT);

    struct dpu_set_t dpu;
    uint32_t each_dpu;
    uint32_t max_join_size = 0;
    DPU_FOREACH(system, dpu, each_dpu) {
        uint32_t join_size = join_res[each_dpu][0].count;
        //std::cout << "Size: " << join_size << std::endl;
        if (join_size > max_join_size) {
            max_join_size = join_size;
        }
    }

    arrow::BufferVector buffers = alloc_buf_vec(max_join_size*sizeof(key_ptr32), NR_DPU);
    get_buf(system, buffers, 0, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    arrow::Result<std::unique_ptr<arrow::Buffer>> map_try = arrow::AllocateBuffer(NR_DPU*OUTER_SIZE*sizeof(uint32_t));
    if (!map_try.ok()) {
        std::cout << "Could not allocate map" << std::endl;
    }
    std::shared_ptr<arrow::Buffer> map = *std::move(map_try);
    uint32_t* map_data = (uint32_t*) map->mutable_data();

    #pragma omp parallel for
    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        key_ptr32* buffer_data = (key_ptr32*) buffers[dpu]->data();
        for (uint32_t i = 0; i < join_res[dpu][0].count; i++) {
            map_data[buffer_data[i].key] = buffer_data[i].ptr;
        }
    }

    return map;
}

int main(void) {
    dpu_set_t system;
    DPU_ASSERT(dpu_alloc(NR_DPU, "sgXferEnable=true", &system));
    init_buffer();
    try {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        std::chrono::steady_clock::time_point begin_init = std::chrono::steady_clock::now();
        DPU_ASSERT(dpu_load(system, "kernel_bjoin", NULL));
        populate_mram(system);
        std::chrono::steady_clock::time_point end_init = std::chrono::steady_clock::now();
        std::cout << "Initial transfer elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_init - begin_init).count()
              << " millisecs." << std::endl;
        
        std::chrono::steady_clock::time_point begin_part = std::chrono::steady_clock::now();
        DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));
        std::chrono::steady_clock::time_point end_part = std::chrono::steady_clock::now();
        std::cout << "Partition elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_part - begin_part).count()
              << " millisecs." << std::endl;

        std::chrono::steady_clock::time_point begin_dist = std::chrono::steady_clock::now();
        uint32_t inner_off = 2*OUTER_SIZE*sizeof(key_ptr32);

        auto join_args = redistribute(system, inner_off);
        std::chrono::steady_clock::time_point end_dist = std::chrono::steady_clock::now();
        std::cout << "Redistribution elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_dist - begin_dist).count()
              << " millisecs." << std::endl;
        
        std::chrono::steady_clock::time_point begin_join = std::chrono::steady_clock::now();
        DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));
        std::chrono::steady_clock::time_point end_join = std::chrono::steady_clock::now();
                std::cout << "Join elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_join - begin_join).count()
              << " millisecs." << std::endl;

        std::chrono::steady_clock::time_point begin_gather = std::chrono::steady_clock::now();
        auto results = get_results(system);
        std::chrono::steady_clock::time_point end_gather = std::chrono::steady_clock::now();
        std::cout << "Gather elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_gather - begin_gather).count()
              << " millisecs." << std::endl;


        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "Host elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " millisecs." << std::endl;

        validate(results);

        output_dpu(system);
    }
    catch (const dpu::DpuError & e) {
        std::cerr << e.what() << std::endl;
    }
    
    return 0;
}