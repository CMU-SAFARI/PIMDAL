#include "shared.cpp"

std::shared_ptr<arrow::Buffer> map;

void populate_mram(dpu_set_t &system) {

    std::vector<std::vector<join_arguments_t>> join_args(NR_DPU, std::vector<join_arguments_t>(1));
    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        join_args[dpu][0].kernel_sel = 0;
        join_args[dpu][0].ptr_inner = dpu*INNER_SIZE;
        join_args[dpu][0].ptr_outer = dpu*OUTER_SIZE;
    }
    dist_vec(system, join_args, 0, "join_args", DPU_XFER_ASYNC);

    dist_table<uint32_t>(system, inner_table, "key", DPU_MRAM_HEAP_POINTER_NAME, 0, DPU_XFER_ASYNC);
    dist_table<uint32_t>(system, outer_table, "key", DPU_MRAM_HEAP_POINTER_NAME, INNER_SIZE*sizeof(key_ptr32), DPU_XFER_ASYNC);
}

dpu_error_t join_callback(struct dpu_set_t rank, uint32_t rank_id, void * args) {

    std::vector<std::vector<join_results_t>> join_res (64, std::vector<join_results_t>(1));

    struct dpu_set_t dpu;
    uint32_t each_dpu;
    DPU_FOREACH(rank, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, (void*) join_res[each_dpu].data()));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, "join_res", 0, join_res[0].size()*sizeof(join_results_t), DPU_XFER_DEFAULT));

    uint32_t max_join_size = 0;
    DPU_FOREACH(rank, dpu, each_dpu) {
        uint32_t join_size = join_res[each_dpu][0].count;
        if (join_size > max_join_size) {
            max_join_size = join_size;
        }
    }

    arrow::BufferVector buffers;
    DPU_FOREACH(rank, dpu, each_dpu) {
        arrow::Result<std::unique_ptr<arrow::Buffer>> selected_try = arrow::AllocateBuffer(max_join_size*sizeof(key_ptr32));
        if (!selected_try.ok()) {
            std::cout << "Cannot allocate buffer!" << std::endl;
            return DPU_ERR_INTERNAL;
        }
        buffers.push_back(*std::move(selected_try));
        DPU_ASSERT(dpu_prepare_xfer(dpu, buffers[each_dpu]->mutable_data()));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, max_join_size*sizeof(key_ptr32), DPU_XFER_DEFAULT));

    // uint32_t* map_data = (uint32_t*) map->mutable_data();

    // DPU_FOREACH(rank, dpu, each_dpu) {
    //     key_ptr32* buffer_data = (key_ptr32*) buffers[each_dpu]->data();
    //     for (uint32_t i = 0; i < join_res[each_dpu][0].count; i++) {
    //         map_data[buffer_data[i].key] = buffer_data[i].ptr;
    //     }
    // }

    return DPU_OK;
}

std::vector<std::vector<join_arguments_t>> redistribute(dpu_set_t &system, uint32_t inner_off,
                                                        uint32_t outer_off, uint32_t inner_size_off,
                                                        uint32_t outer_size_off) {

    std::vector<std::vector<join_arguments_t>> join_args (NR_DPU, std::vector<join_arguments_t>(1));

    arrow::BufferVector part_inner = alloc_buf_vec(INNER_SIZE*sizeof(key_ptr32), NR_DPU);
    arrow::BufferVector part_outer = alloc_buf_vec(OUTER_SIZE*sizeof(key_ptr32), NR_DPU);

    get_buf(system, part_inner, inner_off, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_ASYNC);
    get_buf(system, part_outer, outer_off, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_ASYNC);

    // Copy the sizes of the partitoned tables
    std::vector<std::vector<uint64_t>> inner_sizes(NR_DPU, std::vector<uint64_t>(NR_BUCKETS+1));
    std::vector<std::vector<uint64_t>> outer_sizes(NR_DPU, std::vector<uint64_t>(NR_BUCKETS+1));

    get_vec(system, inner_sizes, inner_size_off, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_ASYNC);
    get_vec(system, outer_sizes, outer_size_off, DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    uint32_t max_inner_size = 0;
    uint32_t max_outer_size = 0;
    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        for (uint32_t src = 0; src < NR_DPU; src++) {
            join_args[dpu][0].n_el_inner += inner_sizes[src][dpu+1] - inner_sizes[src][dpu];
            join_args[dpu][0].n_el_outer += outer_sizes[src][dpu+1] - outer_sizes[src][dpu];
        }

        if (join_args[dpu][0].n_el_inner > max_inner_size) {
            max_inner_size = join_args[dpu][0].n_el_inner;
        }

        if (join_args[dpu][0].n_el_outer > max_outer_size) {
            max_outer_size = join_args[dpu][0].n_el_outer;
        }
    }

    for (uint32_t dpu = 0; dpu < NR_DPU; dpu++) {
        join_args[dpu][0].offset_outer = max_inner_size;
        join_args[dpu][0].offset_inner = max_outer_size;
        join_args[dpu][0].kernel_sel = 1;

        for (uint32_t i = 0; i < NR_BUCKETS+1; i++) {
            inner_sizes[dpu][i] *= sizeof(key_ptr32);
            outer_sizes[dpu][i] *= sizeof(key_ptr32);
        }
    }

    //Calculate offsets for copying to redistribution buffer
    sg_xfer_context_2d sc_args_inner = {.partitions = part_inner, .offset = inner_sizes};
    get_block_t get_block_info_inner = {.f = &get_cpy_ptr_2d, .args = &sc_args_inner, .args_size=(sizeof(sc_args_inner))};

    dpu_sg_xfer_flags_t flag = dpu_sg_xfer_flags_t(DPU_SG_XFER_ASYNC | DPU_SG_XFER_DISABLE_LENGTH_CHECK);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_outer_size*sizeof(key_ptr32),
                                max_inner_size*sizeof(key_ptr32), &get_block_info_inner, flag));

    sg_xfer_context_2d sc_args_outer = {.partitions = part_outer, .offset = outer_sizes};
    get_block_t get_block_info_outer = {.f = &get_cpy_ptr_2d, .args = &sc_args_outer, .args_size=(sizeof(sc_args_outer))};

    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
                                max_outer_size*sizeof(key_ptr32), &get_block_info_outer, flag));

    dist_vec(system, join_args, 0, "join_args", DPU_XFER_ASYNC);

    DPU_ASSERT(dpu_launch(system, DPU_ASYNCHRONOUS));
    DPU_ASSERT(dpu_callback(system, join_callback, NULL, DPU_CALLBACK_ASYNC));

    dpu_sync(system);

    return join_args;
}

int main(void) {
    dpu_set_t system;
    DPU_ASSERT(dpu_alloc(NR_DPU, "sgXferEnable=true", &system));
    init_buffer();
    try {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        DPU_ASSERT(dpu_load(system, "kernel_hjoin", NULL));
        populate_mram(system);

        DPU_ASSERT(dpu_launch(system, DPU_ASYNCHRONOUS));

        // arrow::Result<std::unique_ptr<arrow::Buffer>> map_try = arrow::AllocateBuffer(NR_DPU*OUTER_SIZE*sizeof(uint32_t));
        // if (!map_try.ok()) {
        //    std::cout << "Could not allocate map" << std::endl;
        // }
        // map = *std::move(map_try);

        uint32_t part_off = 0;
        uint32_t match_off = part_off + INNER_SIZE*sizeof(key_ptr32);
        uint32_t part_size_off = match_off + OUTER_SIZE*sizeof(key_ptr32);
        uint32_t match_size_off = part_size_off + (NR_BUCKETS+1)*sizeof(uint64_t);

        auto join_args = redistribute(system, part_off, match_off, part_size_off, match_size_off);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "Host elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " millisecs." << std::endl;

        //validate(map);

        //output_dpu(system);
    }
    catch (const dpu::DpuError & e) {
        std::cerr << e.what() << std::endl;
    }
    
    return 0;
}