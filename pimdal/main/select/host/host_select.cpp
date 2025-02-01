#include "shared.cpp"

#include <map>
#include <mutex>

// Need a map to preserve the order of the chunks
std::map<uint32_t, std::shared_ptr<arrow::Array>> chunks;

std::mutex mutex;

void populate_mram(dpu_set_t &system) {
    select_arguments_t sel_args = {.size = BUFFER_SIZE};
    DPU_ASSERT(dpu_broadcast_to(system, "dpu_args", 0, &sel_args, sizeof(sel_args), DPU_XFER_ASYNC));

    dist_table<int32_t>(system, table, "key", DPU_MRAM_HEAP_POINTER_NAME, 0, DPU_XFER_ASYNC);
}

dpu_error_t select_callback(struct dpu_set_t rank, 
    __attribute((unused)) uint32_t rank_id, void * args) {

    std::vector<std::vector<select_results_t>> sel_res (64, std::vector<select_results_t>(1));

    struct dpu_set_t dpu;
    uint32_t each_dpu;
    DPU_FOREACH(rank, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, (void*) sel_res[each_dpu].data()));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, "dpu_results", 0, sel_res[0].size()*sizeof(select_results_t), DPU_XFER_DEFAULT));

    uint32_t max_sel_size = 0;
    DPU_FOREACH(rank, dpu, each_dpu) {
        uint32_t sel_size = (sel_res[each_dpu][0].count + 7) & (-8);
        if (sel_size > max_sel_size) {
            max_sel_size = sel_size;
        }
    }

    arrow::BufferVector buffers;
    DPU_FOREACH(rank, dpu, each_dpu) {
        arrow::Result<std::unique_ptr<arrow::Buffer>> selected_try = arrow::AllocateBuffer(max_sel_size*sizeof(int32_t));
        if (!selected_try.ok()) {
            std::cout << "Cannot allocate buffer!" << std::endl;
            return DPU_ERR_INTERNAL;
        }
        buffers.push_back(*std::move(selected_try));
        DPU_ASSERT(dpu_prepare_xfer(dpu, buffers[each_dpu]->mutable_data()));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, BUFFER_SIZE*sizeof(key_ptr_t), max_sel_size*sizeof(int32_t), DPU_XFER_DEFAULT));

    DPU_FOREACH(rank, dpu, each_dpu) {
        uint32_t dpu_size = sel_res[each_dpu][0].count;
        auto array_data = arrow::ArrayData::Make(arrow::int32(), dpu_size, {nullptr, buffers[each_dpu]});
        auto array = arrow::MakeArray(array_data);

        // Chunks can be modified by multiple threads simultaneously
        const std::lock_guard<std::mutex> lock(mutex);
        chunks.insert(std::make_pair(rank_id * 64 + each_dpu, array));
    }

    return DPU_OK;
}

std::shared_ptr<arrow::ChunkedArray> concat() {
    arrow::ArrayVector results_vec;
    for(auto const &e : chunks) {
        results_vec.push_back(std::move(e.second));
    }
    auto results = std::make_shared<arrow::ChunkedArray>(results_vec);

    return results;
}

int main(void) {
    dpu_set_t system;
    DPU_ASSERT(dpu_alloc(NR_DPU, "sgXferEnable=true", &system));
    init_buffer();
    try {
        DPU_ASSERT(dpu_load(system, "kernel_select", NULL));
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        populate_mram(system);
        DPU_ASSERT(dpu_launch(system, DPU_ASYNCHRONOUS));

        DPU_ASSERT(dpu_callback(system, select_callback, NULL, DPU_CALLBACK_ASYNC));

        dpu_sync(system);
        auto results = concat();

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "Host elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " millisecs." << std::endl;

        validate(results);
    }
    catch (const dpu::DpuError & e) {
        std::cerr << e.what() << std::endl;
    }
    
    return 0;
}