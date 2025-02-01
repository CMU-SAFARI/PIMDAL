#include "shared.cpp"

void populate_mram(dpu_set_t &system) {
    groupby_arguments_t gb_args = {.size = BUFFER_SIZE, .kernel_sel = aggr_type};
    DPU_ASSERT(dpu_broadcast_to(system, "dpu_args", 0, &gb_args, sizeof(gb_args), DPU_XFER_DEFAULT));

    dist_table<int32_t>(system, table, "key", DPU_MRAM_HEAP_POINTER_NAME, 0, DPU_XFER_DEFAULT);

    if (aggr_type > 0) {
        uint32_t offset = BUFFER_SIZE + (BUFFER_SIZE & 1);
        dist_table<int32_t>(system, table, "val", DPU_MRAM_HEAP_POINTER_NAME, offset*sizeof(int32_t), DPU_XFER_DEFAULT);
    }
}

std::shared_ptr<arrow::Table> get_results(dpu_set_t &system) {
    std::vector<std::vector<groupby_results_t>> gb_res (NR_DPU, std::vector<groupby_results_t>(1));
    arrow::ArrayVector key_chunks;
    arrow::ArrayVector val_chunks;

    get_vec(system, gb_res, 0, "dpu_results", DPU_XFER_DEFAULT);

    struct dpu_set_t dpu;
    uint32_t each_dpu;
    uint32_t max_gb_size = 0;
    uint32_t tot = 0;
    DPU_FOREACH(system, dpu, each_dpu) {
        tot += gb_res[each_dpu][0].count;
        uint32_t gb_size = gb_res[each_dpu][0].count + (gb_res[each_dpu][0].count & 1);
        //std::cout << "Size: " << gb_res[each_dpu][0].count << std::endl;
        if (gb_size > max_gb_size) {
            max_gb_size = gb_size;
        }
    }

    uint32_t offset = BUFFER_SIZE + (BUFFER_SIZE & 1);
    arrow::BufferVector buffers_key = alloc_buf_vec(max_gb_size*sizeof(uint32_t), NR_DPU);
    get_buf(system, buffers_key, offset*2*sizeof(uint32_t), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);

    arrow::BufferVector buffers_val;
    if (aggr_type > 0) {
        buffers_val = alloc_buf_vec(max_gb_size*sizeof(uint32_t), NR_DPU);
        get_buf(system, buffers_val, offset*3*sizeof(uint32_t), DPU_MRAM_HEAP_POINTER_NAME, DPU_XFER_DEFAULT);
    }

    DPU_FOREACH(system, dpu, each_dpu) {
        uint32_t dpu_size = gb_res[each_dpu][0].count;
        auto array_data_key = arrow::ArrayData::Make(arrow::uint32(), dpu_size, {nullptr, buffers_key[each_dpu]});
        auto array_key = arrow::MakeArray(array_data_key);
        key_chunks.push_back(array_key);

        if (aggr_type > 0) {
            auto array_data_val = arrow::ArrayData::Make(arrow::uint32(), dpu_size, {nullptr, buffers_val[each_dpu]});
            auto array_val = arrow::MakeArray(array_data_val);
            val_chunks.push_back(array_val);
        }
    }

    arrow::ChunkedArrayVector data_vec;
    auto schema = arrow::schema({arrow::field("key", arrow::int32(), false)});
    data_vec.push_back(std::make_shared<arrow::ChunkedArray>(key_chunks));

    if (aggr_type > 0) {
        schema = arrow::schema({arrow::field("key", arrow::int32(), false),
                            arrow::field("val", arrow::int32(), false)});
        data_vec.push_back(std::make_shared<arrow::ChunkedArray>(val_chunks));
    }

    return arrow::Table::Make(schema, data_vec);
}

std::shared_ptr<arrow::Table> aggr_host(std::shared_ptr<arrow::Table> results) {
    // Aggregate the received dpu values
    auto options = std::make_shared<arrow::compute::ScalarAggregateOptions>();
    std::shared_ptr<ac::AggregateNodeOptions> aggregate_options;
    if (aggr_type == UNIQUE) {
        aggregate_options = std::make_shared<ac::AggregateNodeOptions>(ac::AggregateNodeOptions{{{"hash_distinct", options, "key", "distinct(key)"}},
                                                                                                {"key"}});
    }
    else {
        aggregate_options = std::make_shared<ac::AggregateNodeOptions>(ac::AggregateNodeOptions{{{"hash_sum", options, "val", "sum(val)"}},
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
        DPU_ASSERT(dpu_load(system, "kernel_haggregate", NULL));
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        std::chrono::steady_clock::time_point begin_init = std::chrono::steady_clock::now();
        populate_mram(system);
        std::chrono::steady_clock::time_point end_init = std::chrono::steady_clock::now();
        std::cout << "Initial transfer elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_init - begin_init).count()
              << " millisecs." << std::endl;

        std::chrono::steady_clock::time_point begin_aggr = std::chrono::steady_clock::now();
        DPU_ASSERT(dpu_launch(system, DPU_SYNCHRONOUS));
        std::chrono::steady_clock::time_point end_aggr = std::chrono::steady_clock::now();
                std::cout << "GroupBy elapsed time: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end_aggr - begin_aggr).count()
                << " millisecs." << std::endl;

        std::chrono::steady_clock::time_point begin_fin = std::chrono::steady_clock::now();
        std::shared_ptr<arrow::Table> results = get_results(system);
        results = aggr_host(results);
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