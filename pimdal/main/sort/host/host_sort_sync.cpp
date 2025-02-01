#include "shared.cpp"

void populate_mram(dpu_set_t &system) {
    kernel_arguments_t sort_args {.kernel_sel = 0, .nr_splits = NR_DPU, .range = uint32_t(-1)};
    DPU_ASSERT(dpu_broadcast_to(system, "kernel_args", 0, (void*) &sort_args,
                                sizeof(kernel_arguments_t), DPU_XFER_DEFAULT));

    dist_table<uint32_t>(system, table, "key", DPU_MRAM_HEAP_POINTER_NAME, 0, DPU_XFER_DEFAULT);
}

/*
system: set of DPUs used for the sorting
nr_elements: output the number of elements each DPU is working on

returns: size of the biggest partition a DPU is working on

Copies the partitioned arrays from the DPUs, merges corresponding partions
and copies data back to the DPUs.
*/
std::vector<std::vector<kernel_arguments_t>> redistribute(dpu_set_t &system, uint32_t part_off, uint32_t size_off,
                                                        uint32_t &part_max_size) {

    std::vector<std::vector<kernel_arguments_t>> sort_args (NR_DPU, std::vector<kernel_arguments_t>(1));
    arrow::BufferVector part = alloc_buf_vec(BUFFER_SIZE*sizeof(key_ptr32), NR_DPU);

    std::vector<std::vector<uint64_t>> sizes(NR_DPU, std::vector<uint64_t>(NR_DPU));
    get_vec(system, sizes, size_off, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    // Copy the partitioned data
    get_buf(system, part, part_off, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    for (uint64_t i = 0; i < NR_DPU; i++) {
        for (uint64_t src = 0; src < NR_DPU; src++) {
            sort_args[i][0].nr_el += sizes[src][i];
        }
        
        if (sort_args[i][0].nr_el > part_max_size) {
            part_max_size = sort_args[i][0].nr_el;
        }
    }

    for (uint64_t i = 0; i < NR_DPU; i++) {
        sort_args[i][0].range = uint32_t(-1) / NR_DPU;
        sort_args[i][0].start = i*sort_args[i][0].range;
        sort_args[i][0].offset_outer = part_max_size;
        sort_args[i][0].kernel_sel = 1;
        sort_args[i][0].nr_splits = 64;
    }

    std::vector<std::vector<uint64_t>> offset (NR_DPU, std::vector<uint64_t>(sizes[0].size()+1));

    //Calculate offsets for copying to redistribution buffer
    for (uint64_t i = 0; i < NR_DPU; i++) {
        for (uint64_t j = 0; j < NR_DPU; j++) {
            offset[i][j+1] = offset[i][j] + sizes[i][j] * sizeof(key_ptr32);
        }
    }

    sg_xfer_context_2d sc_args = {.partitions = part, .offset = offset};
    get_block_t get_block_info = {.f = get_cpy_ptr_2d, .args = &sc_args, .args_size = sizeof(sc_args)};

    dpu_sg_xfer_flags_t flag = dpu_sg_xfer_flags_t(DPU_SG_XFER_DEFAULT | DPU_SG_XFER_DISABLE_LENGTH_CHECK);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
                                part_max_size*sizeof(key_ptr32), &get_block_info, flag));

    dist_vec(system, sort_args, 0, "kernel_args", DPU_XFER_DEFAULT);

    return sort_args;
}

std::shared_ptr<arrow::ChunkedArray> get_results(dpu_set_t &system, uint32_t max_size, std::vector<std::vector<kernel_arguments_t>> sort_args) {

    arrow::ArrayVector results_chunks;

    // Round up to 2
    max_size = max_size + (max_size & 1);

    arrow::BufferVector buffers = alloc_buf_vec(max_size*sizeof(uint32_t), NR_DPU);

    get_buf(system, buffers, 0, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    struct dpu_set_t dpu;
    uint32_t each_dpu;
    DPU_FOREACH(system, dpu, each_dpu) {
        uint32_t dpu_size = sort_args[each_dpu][0].nr_el;
        auto array_data = arrow::ArrayData::Make(arrow::uint32(), dpu_size, {nullptr, buffers[each_dpu]});
        auto array = arrow::MakeArray(array_data);

        results_chunks.push_back(array);
    }

    return std::make_shared<arrow::ChunkedArray>(results_chunks);
}

int main(void) {
    dpu_set_t system;
    DPU_ASSERT(dpu_alloc(NR_DPU, "sgXferEnable=true", &system));
    init_buffer();
    try {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        std::chrono::steady_clock::time_point begin_init = std::chrono::steady_clock::now();
        DPU_ASSERT(dpu_load(system, "kernel_sort", NULL));
        
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

#if PERF > 0
        output_perf(system, true);
#endif

        std::chrono::steady_clock::time_point begin_dist = std::chrono::steady_clock::now();
        uint32_t part_off = 0;
        uint32_t size_off = 2*BUFFER_SIZE * sizeof(key_ptr32);
        uint32_t max_size = 0;
        auto sort_args = redistribute(system, part_off, size_off, max_size);
        std::chrono::steady_clock::time_point end_dist = std::chrono::steady_clock::now();
        std::cout << "Redistribution elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_dist - begin_dist).count()
              << " millisecs." << std::endl;

        std::chrono::steady_clock::time_point begin_sort = std::chrono::steady_clock::now();
        DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));
        std::chrono::steady_clock::time_point end_sort = std::chrono::steady_clock::now();
        std::cout << "Sort elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_sort - begin_sort).count()
              << " millisecs." << std::endl;

        std::chrono::steady_clock::time_point begin_gather = std::chrono::steady_clock::now();
        std::shared_ptr<arrow::ChunkedArray> results = get_results(system, max_size, sort_args);
        std::chrono::steady_clock::time_point end_gather = std::chrono::steady_clock::now();
        std::cout << "Gather elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_gather - begin_gather).count()
              << " millisecs." << std::endl;

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "Host elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " millisecs." << std::endl;

        validate(results);

#if PERF > 0
        output_perf(system, false);
#endif
        
        //output_dpu(system);

    }
    catch (const dpu::DpuError & e) {
        std::cerr << e.what() << std::endl;
    }
    
    return 0;
}