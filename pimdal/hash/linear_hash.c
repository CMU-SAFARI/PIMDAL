#include "datatype.h"
#include "hash_func.h"

#define TABLE_SIZE 256

/*
    @param element element to insert into table
    @param local_table hash table in WRAM
    @param sel which hash function to use

    Inserts elements into hash table using linear hashing probing scheme.
*/
void hash_phase_two(key_ptr32* in, uint32_t n, key_ptr32 *table) {
    for (uint32_t i = 0; i < n; i++) {
        uint32_t pos = hash1(in[i].key);
        for (uint32_t k = 0; k < TABLE_SIZE; k++) {
            if (table[(pos + k) % TABLE_SIZE].key == 0xffffffff) {
                table[(pos + k) % TABLE_SIZE] = in[i];
                break;
            }
        }
    }
}

/*
    @param in WRAM cache of elements to probe
    @param out WRAM cache output of elements found
    @param table hash table in WRAM
    @param n number of input elements

    Probes the table with the input elements using linear probing and writes ones that
    hit to the output. The pointer of the outer relation element becomes the new key.
*/
uint32_t probe_table(key_ptr32 *in, key_ptr32 *out, key_ptr32 *table, uint32_t n) {
    uint32_t out_i = 0;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t pos = hash1(in[i].key);
        for (uint32_t k = 0; k < TABLE_SIZE; k++) {
            if (table[(pos + k) % TABLE_SIZE].key == in[i].key) {
                key_ptr32 joined = {.key = in[i].ptr, .ptr = table[(pos + k) % TABLE_SIZE].ptr};
                out[out_i] = joined;
                out_i++;
                break;
            }
            else if (table[(pos + k) % TABLE_SIZE].key == 0xffffffff) {
                break;
            }
        }
    }

    return out_i;
}