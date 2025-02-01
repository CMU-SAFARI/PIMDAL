#include "shared.cpp"

void populate_mram(dpu_set_t &system) {
    select_arguments_t sel_args = {.size = BUFFER_SIZE};
    DPU_ASSERT(dpu_broadcast_to(system, "dpu_args", 0, &sel_args, sizeof(sel_args), DPU_XFER_DEFAULT));

    dist_table<int32_t>(system, table, "key", DPU_MRAM_HEAP_POINTER_NAME, 0, DPU_XFER_DEFAULT);
}

std::shared_ptr<arrow::ChunkedArray> get_results(dpu_set_t &system) {
    std::vector<std::vector<select_results_t>> sel_res (NR_DPU, std::vector<select_results_t>(1));
    arrow::ArrayVector results_chunks;

    get_vec(system, sel_res, 0, "dpu_results", DPU_XFER_DEFAULT);

    struct dpu_set_t dpu;
    uint32_t each_dpu;
    uint32_t max_sel_size = 0;
    DPU_FOREACH(system, dpu, each_dpu) {
        uint32_t sel_size = (sel_res[each_dpu][0].count + 7) & (-8);
        if (sel_size > max_sel_size) {
            max_sel_size = sel_size;
        }
    }

    arrow::BufferVector buffers = alloc_buf_vec(max_sel_size*sizeof(int32_t), NR_DPU);
    get_buf(system, buffers, BUFFER_SIZE*sizeof(key_ptr_t), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    DPU_FOREACH(system, dpu, each_dpu) {
        uint32_t dpu_size = sel_res[each_dpu][0].count;
        auto array_data = arrow::ArrayData::Make(arrow::int32(), dpu_size, {nullptr, buffers[each_dpu]});
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
        DPU_ASSERT(dpu_load(system, "kernel_select", NULL));
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        std::chrono::steady_clock::time_point begin_init = std::chrono::steady_clock::now();
        populate_mram(system);
        std::chrono::steady_clock::time_point end_init = std::chrono::steady_clock::now();
        std::cout << "Initial transfer elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_init - begin_init).count()
              << " millisecs." << std::endl;

        std::chrono::steady_clock::time_point begin_select = std::chrono::steady_clock::now();
        DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));
        std::chrono::steady_clock::time_point end_select = std::chrono::steady_clock::now();
                std::cout << "Select elapsed time: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end_select - begin_select).count()
                << " millisecs." << std::endl;

        std::chrono::steady_clock::time_point begin_fin = std::chrono::steady_clock::now();
        std::shared_ptr<arrow::ChunkedArray> results = get_results(system);
        std::chrono::steady_clock::time_point end_fin = std::chrono::steady_clock::now();
        std::cout << "Final transfer elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_fin - begin_fin).count()
              << " millisecs." << std::endl;
        //output_int(results);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "Host elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " millisecs." << std::endl;

        validate(results);

        //output_dpu(system);
#if PERF > 0
        output_perf(system);
#endif

    }
    catch (const dpu::DpuError & e) {
        std::cerr << e.what() << std::endl;
    }
    
    return 0;
}