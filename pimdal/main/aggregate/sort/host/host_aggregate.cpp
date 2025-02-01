#include <mutex>

#include "shared.cpp"

arrow::ArrayVector chunks_key;
arrow::ArrayVector chunks_val;

std::mutex mutex;

void populate_mram(dpu_set_t &system) {
    groupby_arguments_t gb_args = {.size = BUFFER_SIZE, .kernel_sel = aggr_type};
    DPU_ASSERT(dpu_broadcast_to(system, "dpu_args", 0, &gb_args, sizeof(gb_args), DPU_XFER_ASYNC));

    dist_table<int32_t>(system, table, "key", DPU_MRAM_HEAP_POINTER_NAME, 0, DPU_XFER_ASYNC);

    if (aggr_type > 0) {
        dist_table<int32_t>(system, table, "val", DPU_MRAM_HEAP_POINTER_NAME, BUFFER_SIZE*sizeof(int32_t), DPU_XFER_ASYNC);
    }
}

dpu_error_t groupby_callback(struct dpu_set_t rank, 
    __attribute((unused)) uint32_t rank_id, void * args) {

    std::vector<std::vector<groupby_results_t>> gb_res (64, std::vector<groupby_results_t>(1));

    struct dpu_set_t dpu;
    uint32_t each_dpu;
    DPU_FOREACH(rank, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, (void*) gb_res[each_dpu].data()));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, "dpu_results", 0, gb_res[0].size()*sizeof(groupby_results_t), DPU_XFER_DEFAULT));

    uint32_t max_gb_size = 0;
    DPU_FOREACH(rank, dpu, each_dpu) {
        uint32_t gb_size = gb_res[each_dpu][0].count + (gb_res[each_dpu][0].count & 1);
        if (gb_size > max_gb_size) {
            max_gb_size = gb_size;
        }
    }

    arrow::BufferVector buffers_key;
    DPU_FOREACH(rank, dpu, each_dpu) {
        arrow::Result<std::unique_ptr<arrow::Buffer>> gb_try = arrow::AllocateBuffer(max_gb_size*sizeof(int32_t));
        if (!gb_try.ok()) {
            std::cout << "Cannot allocate buffer!" << std::endl;
            return DPU_ERR_INTERNAL;
        }
        buffers_key.push_back(*std::move(gb_try));
        DPU_ASSERT(dpu_prepare_xfer(dpu, buffers_key[each_dpu]->mutable_data()));
    }
    DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, max_gb_size*sizeof(int32_t), DPU_XFER_DEFAULT));

    DPU_FOREACH(rank, dpu, each_dpu) {
        uint32_t dpu_size = gb_res[each_dpu][0].count;
        auto array_data = arrow::ArrayData::Make(arrow::int32(), dpu_size, {nullptr, buffers_key[each_dpu]});
        auto array = arrow::MakeArray(array_data);

        // Chunks can be modified by multiple threads simultaneously
        const std::lock_guard<std::mutex> lock(mutex);
        chunks_key.push_back(array);
    }

    if (aggr_type > 0) {
        arrow::BufferVector buffers_val;
        DPU_FOREACH(rank, dpu, each_dpu) {
            arrow::Result<std::unique_ptr<arrow::Buffer>> gb_try = arrow::AllocateBuffer(max_gb_size*sizeof(int32_t));
            if (!gb_try.ok()) {
                std::cout << "Cannot allocate buffer!" << std::endl;
                return DPU_ERR_INTERNAL;
            }
            buffers_val.push_back(*std::move(gb_try));
            DPU_ASSERT(dpu_prepare_xfer(dpu, buffers_val[each_dpu]->mutable_data()));
        }
        DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, BUFFER_SIZE*sizeof(uint32_t), max_gb_size*sizeof(int32_t), DPU_XFER_DEFAULT));

        DPU_FOREACH(rank, dpu, each_dpu) {
            uint32_t dpu_size = gb_res[each_dpu][0].count;
            auto array_data = arrow::ArrayData::Make(arrow::int32(), dpu_size, {nullptr, buffers_val[each_dpu]});
            auto array = arrow::MakeArray(array_data);

            // Chunks can be modified by multiple threads simultaneously
            const std::lock_guard<std::mutex> lock(mutex);
            chunks_val.push_back(array);
        }
    }

    return DPU_OK;
}

std::shared_ptr<arrow::Table> concat() {

    arrow::ChunkedArrayVector data_vec;
    auto schema = arrow::schema({arrow::field("key", arrow::int32(), false)});
    data_vec.push_back(std::make_shared<arrow::ChunkedArray>(chunks_key));

    if (aggr_type > 0) {
        schema = arrow::schema({arrow::field("key", arrow::int32(), false),
                            arrow::field("val", arrow::int32(), false)});
        data_vec.push_back(std::make_shared<arrow::ChunkedArray>(chunks_val));
    }

    auto results = arrow::Table::Make(schema, data_vec);

    // Aggregate the received dpu values
    auto options = std::make_shared<arrow::compute::ScalarAggregateOptions>();
    std::shared_ptr<ac::AggregateNodeOptions> aggregate_options;
    if (aggr_type == UNIQUE) {
        aggregate_options = std::make_shared<ac::AggregateNodeOptions>(ac::AggregateNodeOptions{{{"hash_distinct", options, "key"}},
                                                                                                {"key"}});
    }
    else {
        aggregate_options = std::make_shared<ac::AggregateNodeOptions>(ac::AggregateNodeOptions{{{"hash_sum", options, "val", "sum"}},
                                                                                                {"key"}});
    }

    auto order_by_options = ac::OrderByNodeOptions({{arrow::compute::SortKey("key")}});
    auto plan = ac::Declaration::Sequence({{"table_source", ac::TableSourceNodeOptions(results)},
                                        {"aggregate", *aggregate_options},
                                        {"order_by", order_by_options}});

    ac::QueryOptions query_options;
    query_options.use_threads = true;
    auto result = ac::DeclarationToTable(plan, query_options);
    auto result_data = result.ValueOrDie();
    
    return result_data;
}

int main(void) {
    dpu_set_t system;
    DPU_ASSERT(dpu_alloc(NR_DPU, "sgXferEnable=true", &system));
    init_buffer();
    try {
        DPU_ASSERT(dpu_load(system, "kernel_saggregate", NULL));
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        populate_mram(system);
        DPU_ASSERT(dpu_launch(system, DPU_ASYNCHRONOUS));

        DPU_ASSERT(dpu_callback(system, groupby_callback, NULL, DPU_CALLBACK_ASYNC));

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