#include "shared.cpp"

void populate_mram(dpu_set_t &system) {
    std::vector<std::vector<join_arguments_t>> join_args(NR_DPU, std::vector<join_arguments_t>(1));
    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        join_args[dpu][0].kernel_sel = 0;
        join_args[dpu][0].ptr_inner = dpu*INNER_SIZE;
        join_args[dpu][0].ptr_outer = dpu*OUTER_SIZE;
        join_args[dpu][0].nr_splits = NR_DPU;
        join_args[dpu][0].range = OUTER_RANGE*NR_DPU*INNER_SIZE;
    }
    dist_vec(system, join_args, 0, "join_args", DPU_XFER_DEFAULT);

    dist_table<uint32_t>(system, inner_table, "key", DPU_MRAM_HEAP_POINTER_NAME, 0, DPU_XFER_DEFAULT);

    dist_table<uint32_t>(system, outer_table, "key", DPU_MRAM_HEAP_POINTER_NAME, INNER_SIZE*sizeof(key_ptr32), DPU_XFER_DEFAULT);
}

/*
system: set of DPUs used for the sorting
nr_elements: output the number of elements each DPU is working on

returns: size of the biggest partition a DPU is working on

Copies the partitioned arrays from the DPUs, merges corresponding partions
and copies data back to the DPUs.
*/
std::vector<std::vector<join_arguments_t>> redistribute(dpu_set_t &system, uint32_t inner_off, uint32_t outer_off,
                                                        uint32_t inner_size_off, uint32_t outer_size_off,
                                                        uint32_t &inner_max_size, uint32_t &outer_max_size) {

    std::vector<std::vector<join_arguments_t>> join_args (NR_DPU, std::vector<join_arguments_t>(1));

    arrow::BufferVector inner_part = alloc_buf_vec(INNER_SIZE*sizeof(key_ptr32), NR_DPU);
    arrow::BufferVector outer_part = alloc_buf_vec(OUTER_SIZE*sizeof(key_ptr32), NR_DPU);

    std::vector<std::vector<uint64_t>> inner_sizes(NR_DPU, std::vector<uint64_t>(NR_DPU));
    std::vector<std::vector<uint64_t>> outer_sizes(NR_DPU, std::vector<uint64_t>(NR_DPU));
    //system.copy(indices, "indices_glob");
    get_vec(system, inner_sizes, inner_size_off, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    get_vec(system, outer_sizes, outer_size_off, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    // Copy the partitioned data
    get_buf(system, inner_part, inner_off, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    get_buf(system, outer_part, outer_off, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    uint32_t size = 0;
    for (uint64_t i = 0; i < NR_DPU; i++) {
        for (uint64_t src = 0; src < NR_DPU; src++) {
            join_args[i][0].n_el_inner += inner_sizes[src][i];
            join_args[i][0].n_el_outer += outer_sizes[src][i];
        }
        
        if (join_args[i][0].n_el_inner > inner_max_size) {
            inner_max_size = join_args[i][0].n_el_inner;
        }

        if (join_args[i][0].n_el_outer > outer_max_size) {
            outer_max_size = join_args[i][0].n_el_outer;
        }
    }

    for (uint64_t i = 0; i < NR_DPU; i++) {
        join_args[i][0].range = OUTER_RANGE*INNER_SIZE;
        join_args[i][0].start = i*OUTER_RANGE*INNER_SIZE;
        join_args[i][0].offset_outer = inner_max_size;
        join_args[i][0].kernel_sel = 1;
    }

    std::vector<std::vector<uint64_t>> inner_offset (NR_DPU, std::vector<uint64_t>(inner_sizes[0].size()+1));
    std::vector<std::vector<uint64_t>> outer_offset (NR_DPU, std::vector<uint64_t>(outer_sizes[0].size()+1));

    //Calculate offsets for copying to redistribution buffer

    for (uint64_t i = 0; i < NR_DPU; i++) {
        for (uint64_t j = 0; j < NR_DPU; j++) {
            inner_offset[i][j+1] = inner_offset[i][j] + inner_sizes[i][j]*sizeof(key_ptr32);
            outer_offset[i][j+1] = outer_offset[i][j] + outer_sizes[i][j]*sizeof(key_ptr32);
        }
    }

    sg_xfer_context_2d sc_args = {.partitions = inner_part, .offset = inner_offset};
    get_block_t get_block_info = {.f = &get_cpy_ptr_2d, .args = &sc_args, .args_size = sizeof(sc_args)};

    dpu_sg_xfer_flags_t flag = dpu_sg_xfer_flags_t(DPU_SG_XFER_DEFAULT | DPU_SG_XFER_DISABLE_LENGTH_CHECK);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
                                inner_max_size*sizeof(key_ptr32), &get_block_info, flag));

    sc_args.partitions = outer_part;
    sc_args.offset = outer_offset;
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, inner_max_size*sizeof(key_ptr32),
                                outer_max_size*sizeof(key_ptr32), &get_block_info, flag));

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
        DPU_ASSERT(dpu_load(system, "kernel_join", NULL));
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
        output_perf_part(system);
#endif

        std::chrono::steady_clock::time_point begin_dist = std::chrono::steady_clock::now();
        uint32_t inner_off = 0;
        uint32_t outer_off = inner_off + INNER_SIZE * sizeof(key_ptr32);
        uint32_t inner_size_off = outer_off + (INNER_SIZE + 2*OUTER_SIZE)*sizeof(key_ptr32);
        uint32_t outer_size_off = inner_size_off + NR_DPU * sizeof(uint64_t);
        uint32_t inner_max_size = 0;
        uint32_t outer_max_size = 0;
        redistribute(system, inner_off, outer_off, inner_size_off, outer_size_off, inner_max_size, outer_max_size);
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
        //output_dpu(system);

#if PERF > 0
        output_perf_merge(system);
#endif

    }
    catch (const dpu::DpuError & e) {
        std::cerr << e.what() << std::endl;
    }
    
    return 0;
}