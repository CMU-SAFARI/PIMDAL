#ifndef _HASHJOIN_H_
#define _HASHJOIN_H_

#include "datatype.h"

typedef struct
{
    uint32_t in_ptr; // Input elements in MRAM
    uint32_t size; // Number of input elements
    uint32_t shift; // Maximum shift for selecting hash bits
    uint32_t table_ptr; // Table in MRAM
    uint32_t table_size; // Table size in number of elements
} hash_arguments_t;

typedef struct
{
    uint32_t in_ptr; // Partitioned input elements in MRAM
    uint32_t filter_ptr; // Bloom filters in MRAM
    uint32_t part_sizes; // Number of elements in each partition
    uint32_t part_n; // Number of partitions
} filter_arguments_t;

typedef struct
{
    uint32_t in_ptr; // Input elements in MRAM
    uint32_t size; // Number of input elements
    uint32_t shift; // Maximum shift for selecting hash bits
    uint32_t part_ptr; // Partitions output in MRAM
    uint32_t part_sizes; // Sizes of the partitions
    uint32_t part_n; // Number of partitions
} part_arguments_t;

typedef struct
{
    uint32_t filter_ptr; // Bloom filters
    uint32_t size; // Number of input elements
    uint32_t part_ptr; // Partitioned input elements in MRAM
    uint32_t out_ptr; // Matched elements output in MRAM
    uint32_t match_sizes; // Partition sizes of matched elements
    uint32_t part_sizes; // Partition sizes of input
    uint32_t part_n; // Number of partitions
} match_arguments_t;

typedef struct
{
    uint32_t table_ptr; // Hash table in MRAM
    uint32_t size; // Size of hash table
    uint32_t in_ptr; // Input elements in MRAM
    uint32_t out_ptr; // Output of merged elements in MRAM
    uint32_t part_sizes; // Partition sizes of input
    uint32_t part_n; // Number of partitions
} merge_arguments_t;

typedef struct
{
    uint32_t out_n;
} merge_results_t;

/*
    Create a hash table from the input elements.
*/
int hash_kernel(hash_arguments_t *hash_args);

/*
    Create a bloom filter from the input elements.
*/
int filter_kernel(filter_arguments_t *filter_args);

/*
    Partition the input elements.
*/
int part_kernel(part_arguments_t *part_args);

/*
    Probe the bloom filter with the input elements.
*/
int match_kernel(match_arguments_t *match_args, merge_results_t *join_res);

/*
    Hash join two relations using a previously created hash table.
*/
int merge_kernel(merge_arguments_t *join_args, merge_results_t *join_res);

#endif