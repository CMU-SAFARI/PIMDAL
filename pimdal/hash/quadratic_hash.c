#include "datatype.h"
#include "hash_func.c"

#define TABLE_SIZE 256

/*
    @param element element to insert into table
    @param local_table hash table in WRAM
    @param sel which hash function to use

    Inserts elements into hash table using quadratic hashing probing scheme.
*/
void hash_phase_two(key_ptr32* in, uint32_t n, key_ptr32 *table) {
    for (uint32_t i = 0; i < n; i++) {
        uint32_t pos = hash1(in[i].key);
        while (table[pos % TABLE_SIZE].key != 0) {
            pos++;
        }

        table[pos % TABLE_SIZE] = in[i];
    }
}

/*
    @param in WRAM cache of elements to probe
    @param out WRAM cache output of elements found
    @param table hash table in WRAM
    @param n number of input elements

    Probes the table with the input elements using quadratic probing and writes ones
    that hit to the output.
*/
uint32_t probe_table(key_ptr32 *in, key_ptr32 *out, key_ptr32 *table, uint32_t n) {
    uint32_t out_i = 0;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t pos = hash1(in[i].key);
        while (table[pos % TABLE_SIZE].key != 0) {
            if (table[pos % TABLE_SIZE].key == in[i].key) {
                key_ptr32 joined = {.key = in[i].key, .ptr = table[pos % TABLE_SIZE].key};
                out[out_i] = joined;
                out_i++;
                break;
            }
            pos++;
        }
    }

    return out_i;
}