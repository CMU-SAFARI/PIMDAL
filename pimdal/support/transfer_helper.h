#include <dpu>

#include <arrow/api.h>
#include <iostream>

#ifndef NR_DPU
#define NR_DPU 4
#endif

typedef struct sg_xfer_context_table {
    std::shared_ptr<arrow::Array> column;
    uint32_t type_size;
} sg_xfer_context_table;

typedef struct sg_xfer_context_buf {
    std::shared_ptr<arrow::Buffer> buffer;
    std::vector<uint64_t> offset;
} sg_xfer_context_buf;

std::vector<sg_xfer_context_table> sc_args_table;
std::vector<sg_xfer_context_buf> sc_args_buf;
std::vector<get_block_t> get_block_info;

/*
Print vector

@param out vector to print
*/
template <typename T>
void output(std::vector<T> out) {
    for (auto & e: out) {
        std::cout << +e << std::endl;
    }
}

/*
Print buffer consisting of int data

@param out arrow Buffer of int32 elements
*/
void output_int(std::shared_ptr<arrow::Buffer> out) {
    int32_t* data = (int32_t*) out->data();
    size_t n_elements = out->size() / sizeof(int32_t);
    for (size_t i = 0; i < n_elements; i++) {
        std::cout << data[i] << std::endl;
    }
}

/*
Print buffer consisting of int data

@param out arrow Buffer of int64 elements
*/
void output_int64(std::shared_ptr<arrow::Buffer> out) {
    int64_t* data = (int64_t*) out->data();
    size_t n_elements = out->size() / sizeof(int64_t);
    for (size_t i = 0; i < n_elements; i++) {
        std::cout << data[i] << std::endl;
    }
}

/*
Print keys in buffer constisting of key, value pairs

@param out arrow Buffer of uint32 keys and values
*/
void output_key(std::shared_ptr<arrow::Buffer> out) {
    uint32_t* data = (uint32_t*) out->data();
    size_t n_elements = out->size() / sizeof(uint32_t);
    for (size_t i = 0; i < n_elements; i++) {
        if (i % 2 == 0) {
            std::cout << data[i] << std::endl;
        }
    }
}

/*
Distribute a contiguous arrow::Buffer to the DPUs in the set.

@param system dpus to send to
@param buffer arrow Buffer to distribute
@param offset offset from the dpu destination symbol
@param size_dpu size to send to each dpu
@param DstSymbol dpu destination symbol
@param flag options to the transfer
*/
void dist_buf(dpu_set_t &system, std::shared_ptr<arrow::Buffer> &buffer, uint32_t offset, uint32_t size_dpu, const std::string &DstSymbol, dpu_xfer_flags_t flag) {
    struct dpu_set_t dpu;
    unsigned dpuIdx;
    DPU_FOREACH (system, dpu, dpuIdx) {
        std::shared_ptr<arrow::Buffer> slice = arrow::SliceBuffer(buffer, (uint64_t) dpuIdx*size_dpu, size_dpu);
        DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)slice->data()));
    }
    DPU_ASSERT(dpu_push_xfer(system, DPU_XFER_TO_DPU, DstSymbol.c_str(), offset, size_dpu, flag));
}

void copy_buf(dpu_set_t &system, std::shared_ptr<arrow::Buffer> &buffer, uint32_t offset, const std::string &DstSymbol, dpu_xfer_flags_t flag) {
    uint32_t size = buffer->size();
    DPU_ASSERT(dpu_broadcast_to(system, DstSymbol.c_str(), offset, (void*) buffer->mutable_data(), size, flag));
}

bool get_table_ptr (struct sg_block_info *out, uint32_t dpu_index,
                              uint32_t block_index, void *args) {

    if (block_index >= 1) {
        return false;
    }

    sg_xfer_context_table *sc_args = reinterpret_cast<sg_xfer_context_table*>(args);
    auto array = sc_args->column;

    uint64_t start;
    uint64_t length;
    if (array->length() % NR_DPU == 0) {
        start = dpu_index * (array->length()/NR_DPU);
        length = array->length()/NR_DPU;
    }
    else {
        start = dpu_index * (array->length()/NR_DPU + 1);
        if (dpu_index < NR_DPU - 1) {
            length = array->length()/NR_DPU + 1;
        }
        else {
            length = array->length() % (array->length()/NR_DPU + 1);
        }
    }

    out->length = length * sc_args->type_size;
    out->addr = (uint8_t*) array->data()->GetMutableValues<uint8_t>(1, start*sc_args->type_size);
    //printf("Addr %u %u: %p %u\n", dpu_index, block_index, out->addr, out->length);

    return true;
}

/*
Split the column of a table between all DPUs using scatter transfers.

@param system dpus to send to
@param table arrow Table to distribute
@param column name of the column to distribute
@param DstSymbol dpu destination symbol
@param offset offset from the dpu destination symbol
@param flag options for the transfer
*/
void scatter_table(dpu_set_t system, std::shared_ptr<arrow::Table> table,
                std::string column, const std::string &DstSymbol, uint32_t offset,
                dpu_sg_xfer_flags_t flag) {

    auto col_data = table->GetColumnByName(column)->chunk(0);

    auto dtype = col_data->type();
    uint32_t type_size = dtype->layout().buffers[1].byte_width;
    
    sc_args_table.push_back(sg_xfer_context_table({.column = col_data, .type_size = type_size}));
    get_block_info.push_back(get_block_t({.f = get_table_ptr, .args = &sc_args_table.back(), .args_size = sizeof(sg_xfer_context_table)}));

    uint32_t length = ((col_data->length()/NR_DPU + 1) * type_size + 7) & (-8);

    flag = dpu_sg_xfer_flags_t(flag | DPU_SG_XFER_DISABLE_LENGTH_CHECK);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_TO_DPU, DstSymbol.c_str(), offset,
                                length, &get_block_info.back(), flag));
}

/*
Copy the column of a table to all DPUs.

@param system dpus to send to
@param table arrow Table to distribute
@param column name of the column to distribute
@param DstSymbol dpu destination symbol
@param offset offset from the dpu destination symbol
@param flag options for the transfer
*/
void copy_table(dpu_set_t &system, std::shared_ptr<arrow::Table> table,
                std::string column, const std::string &DstSymbol, uint32_t offset,
                dpu_xfer_flags_t flag) {
    
    auto col_array = table->GetColumnByName(column)->chunk(0);

    auto dtype = col_array->type();
    uint32_t type_size = dtype->layout().buffers[1].byte_width;

    uint32_t length = col_array->length() * type_size;
    uint8_t* data = col_array->data()->GetMutableValues<uint8_t>(1, 0);

    // Copy the data to a new buffer for padding
    uint32_t length_pad = (length + 7) & (-8);
    arrow::Result<std::unique_ptr<arrow::Buffer>> buffer_try = arrow::AllocateBuffer(length_pad);
    std::shared_ptr<arrow::Buffer> buffer = *std::move(buffer_try);

    std::memcpy(buffer->mutable_data(), data, length);

    DPU_ASSERT(dpu_broadcast_to(system, DstSymbol.c_str(), offset, (void*) buffer->mutable_data(), length_pad, flag));
}


/*
Split the column of a table between all DPUs using parallel transfers.

@param system dpus to send to
@param table arrow Table to distribute
@param column name of the column to distribute
@param DstSymbol dpu destination symbol
@param offset offset from the dpu destination symbol
@param flag options for the transfer
*/
template <typename T>
void dist_table(dpu_set_t system, std::shared_ptr<arrow::Table> table,
                std::string column, const std::string &DstSymbol, uint32_t offset,
                dpu_xfer_flags_t flag) {
    // Detect buffer size in bytes
    auto col_data = table->GetColumnByName(column)->chunk(0);
    auto dtype = col_data->type();

    uint64_t length = (col_data->length() / NR_DPU);

    struct dpu_set_t dpu;
    unsigned dpuIdx;
    DPU_FOREACH (system, dpu, dpuIdx) {
        std::shared_ptr<arrow::Array> slice = col_data->Slice((uint64_t) dpuIdx * length, length);

        DPU_ASSERT(dpu_prepare_xfer(dpu, (void *)slice->data()->GetValues<T>(1)));
    }
    DPU_ASSERT(dpu_push_xfer(system, DPU_XFER_TO_DPU, DstSymbol.c_str(), offset, length*sizeof(T), flag));

}

void get_buf(dpu_set_t &system, arrow::BufferVector &buffer, uint32_t offset, const std::string &SrcSymbol, dpu_xfer_flags_t flag) {
    struct dpu_set_t dpu;
    unsigned dpuIdx;
    unsigned size = buffer[0]->size();
    DPU_FOREACH (system, dpu, dpuIdx) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, (void*)buffer[dpuIdx]->data()));
    }
    DPU_ASSERT(dpu_push_xfer(system, DPU_XFER_FROM_DPU, SrcSymbol.c_str(), offset, size, flag));
}

bool get_buf_ptr (struct sg_block_info *out, uint32_t dpu_index,
                    uint32_t block_index, void *args) {

    if (block_index >= 1) {
        return false;
    }

    sg_xfer_context_buf *sc_args = reinterpret_cast<sg_xfer_context_buf*>(args);

    out->length = (sc_args->offset)[dpu_index + 1] - (sc_args->offset)[dpu_index];
    out->addr = (sc_args->buffer)->mutable_data() + (sc_args->offset)[dpu_index];

    return true;
}

void collect_buf(dpu_set_t &system, std::shared_ptr<arrow::Buffer> buffer, uint32_t offset, const std::string &SrcSymbol,
                 uint32_t length, std::vector<uint64_t> buf_offset, dpu_sg_xfer_flags_t flag) {
    
    sc_args_buf.push_back(sg_xfer_context_buf({.buffer = buffer, .offset = buf_offset}));
    get_block_info.push_back(get_block_t({.f = get_buf_ptr, .args = &sc_args_buf.back(), .args_size = sizeof(sg_xfer_context_buf)}));

    flag = dpu_sg_xfer_flags_t(flag | DPU_SG_XFER_DISABLE_LENGTH_CHECK);
    DPU_ASSERT(dpu_push_sg_xfer(system, DPU_XFER_FROM_DPU, SrcSymbol.c_str(), offset,
                                length, &get_block_info.back(), flag));
}

template <typename T>
void dist_vec(dpu_set_t &system, std::vector<std::vector<T>> &buffer, uint32_t offset, const std::string &DstSymbol, dpu_xfer_flags_t flag) {
    struct dpu_set_t dpu;
    unsigned dpuIdx;
    unsigned size = buffer[0].size();
    DPU_FOREACH (system, dpu, dpuIdx) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, (void*)buffer[dpuIdx].data()));
    }
    DPU_ASSERT(dpu_push_xfer(system, DPU_XFER_TO_DPU, DstSymbol.c_str(), offset, size*sizeof(T), flag));
}

template <typename T>
void get_vec(dpu_set_t &system, std::vector<std::vector<T>> &buffer, uint32_t offset, const std::string &Src, dpu_xfer_flags_t flag) {
    struct dpu_set_t dpu;
    unsigned dpuIdx;
    DPU_FOREACH (system, dpu, dpuIdx) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, (void*)buffer[dpuIdx].data()));
    }
    DPU_ASSERT(dpu_push_xfer(system, DPU_XFER_FROM_DPU, Src.c_str(), offset, buffer[0].size()*sizeof(T), flag));
}

typedef struct sg_xfer_context_2d {
    arrow::BufferVector &partitions;
    std::vector<std::vector<uint64_t>> &offset;
} sg_xfer_context_2d;

static bool get_cpy_ptr_2d(struct sg_block_info *out, uint32_t dpu_index, uint32_t block_index,
                 void *args) {

    if (block_index >= NR_DPU) {
        return false;
    }

    sg_xfer_context_2d *sc_args = reinterpret_cast<sg_xfer_context_2d*>(args);

    out->length = (sc_args->offset)[block_index][dpu_index + 1] - (sc_args->offset)[block_index][dpu_index];
    out->addr = (sc_args->partitions)[block_index]->mutable_data() + (sc_args->offset)[block_index][dpu_index];
    //printf("Addr %u %u: %p %u\n", dpu_index, block_index, out->addr, out->length);

    return true;
}

typedef struct sg_xfer_context_1d {
    std::shared_ptr<arrow::Buffer> partitions;
    std::vector<uint64_t> offset;
} sg_xfer_context_1d;

static bool get_cpy_ptr_1d(struct sg_block_info *out, uint32_t dpu_index, uint32_t block_index,
                 void *args) {

    if (block_index >= 1) {
        return false;
    }

    sg_xfer_context_1d *sc_args = (sg_xfer_context_1d *) args;

    out->length = (sc_args->offset)[dpu_index + 1] - (sc_args->offset)[dpu_index];
    out->addr = (sc_args->partitions)->mutable_data() + (sc_args->offset)[dpu_index];
    //printf("Addr %u %u: %p %u\n", dpu_index, block_index, out->addr, out->length);

    return true;
}

/*
Allocate a vector of arrow::Buffers

@param size the size of each individual buffer
@param len the number of buffers

@returns: the allocated vector of buffers
*/
arrow::BufferVector alloc_buf_vec(uint64_t size, uint64_t len) {
    arrow::BufferVector buffer_vec;
    for (uint32_t i = 0; i < len; i++) {
        arrow::Result<std::unique_ptr<arrow::Buffer>> buffer_try = arrow::AllocateBuffer(size);
        if (!buffer_try.ok()) {
            std::cout << "Could not allocate buffer!" << std::endl;
        }
        std::shared_ptr<arrow::Buffer> part_dpu = *std::move(buffer_try);
        buffer_vec.push_back(part_dpu);
    }

    return buffer_vec;
}

/*
Compare if two arrow tables consist of the same values.

@param results input arrow Table to compare
@param comparison arrow Table to compare with
*/
bool verify_table(std::shared_ptr<arrow::Table> results,
                  std::shared_ptr<arrow::Table> comparison) {
    
    if (results->num_columns() != comparison->num_columns()) {
        return false;
    }

    for (uint32_t column = 0; column < results->num_columns(); column++) {
        if (!results->column(column)->Equals(comparison->column(column))){
            return false;
        }
    }

    return true;
}

/*
Print dpu output

@param system set of dpus to get printf output from
*/
void output_dpu(dpu_set_t &system) {
    dpu_set_t dpu;
    unsigned int each_dpu = 0;
    printf("Display DPU Logs\n");
    DPU_FOREACH(system, dpu)
    {
        printf("DPU#%d:\n", each_dpu);
        DPU_ASSERT(dpu_log_read(dpu, stdout));
        each_dpu++;
    }
}