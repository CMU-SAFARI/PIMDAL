#include <unordered_map>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/compute/api.h>
#include <parquet/arrow/reader.h>

int32_t date_to_int(std::string input) {

    struct tm date;
    strptime(input.c_str(), "%Y-%m-%d", &date);

    int32_t val = date.tm_mday + ((date.tm_mon+1) << 5) + (date.tm_year << 9);

    return val;
}

arrow::Status parquet_to_table(std::string table_name,
                               std::shared_ptr<arrow::Table> &table,
                               std::vector<int32_t> sel_col) {

    std::string path = "../../../main/tpc_h/data/";
    path = path + table_name + "/part.0.parquet";

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    std::shared_ptr<arrow::io::RandomAccessFile> input;
    ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(path));

    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    ARROW_RETURN_NOT_OK(parquet::arrow::OpenFile(input, pool, &arrow_reader));

    ARROW_RETURN_NOT_OK(arrow_reader->ReadTable(sel_col, &table));

    return arrow::Status::OK();
}