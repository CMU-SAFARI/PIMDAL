#include "datatype.h"
#include "hash_func.c"

#define TABLE_2_SIZE 64
#define NR_SUBTABLES 4

/*
    @param element element to insert into table
    @param local_table hash table in WRAM
    @param sel which hash function to use

    Tries to insert the element into the hash table.
*/
key_ptr32 insert_table(key_ptr32 element, key_ptr32 *local_table, uint32_t sel) {

    uint32_t hash = hash_func[sel](element.key);
    hash = hash % TABLE_2_SIZE;

    if (local_table[sel*TABLE_2_SIZE + hash].key == 0) {
        local_table[sel*TABLE_2_SIZE + hash] = element;
        return (key_ptr32) {.key = 0, .ptr = 0};
    }
    else if (local_table[sel*TABLE_2_SIZE + hash].key == element.key) {
        return (key_ptr32) {.key = 0, .ptr = 0};
    }

    key_ptr32 evicted = local_table[sel*TABLE_2_SIZE + hash];
    local_table[sel*TABLE_2_SIZE + hash] = element;

    return evicted;
}

/*
    @param in WRAM cache of input elements
    @param n number of elements
    @param table hash tables used

    Insert elements into the hash table using cuckoo hashing scheme.
*/
void hash_phase_two(key_ptr32* in, uint32_t n, key_ptr32 *table) {
    for (uint32_t i = 0; i < n; i++) {
        key_ptr32 el = in[i];
        uint32_t sel = 0;
        //uint32_t success = 0;
        for (uint32_t iter = 0; iter < 256; iter++) {
            el = insert_table(el, table, sel);
            sel = (sel + 1) % NR_SUBTABLES;
            if (el.key == 0) {
                //success = 1;
                break;
            }
        }
        //if (!success) {
            //printf("Miss %u size: %u\n", el.key, n);
        //}
    }
}

/*
    @param in WRAM cache of elements to probe
    @param out WRAM cache output of elements found
    @param table hash table in WRAM
    @param n number of input elements

    Probes the table with the input elements using cuckoo hashing and writes ones that
    hit to the output.
*/
uint32_t probe_table(key_ptr32 *in, key_ptr32 *out, key_ptr32 *table, uint32_t n) {
    uint32_t out_i = 0;

    for (uint32_t i = 0; i < n; i++) {
        //uint32_t hit = 0;
        //uint32_t probed[4];
        for (uint32_t subtable = 0; subtable < NR_SUBTABLES; subtable++) {
            uint32_t h2 = hash_func[subtable](in[i].key);
            h2 = h2 % TABLE_2_SIZE;

            if (table[subtable*TABLE_2_SIZE + h2].key == in[i].key) {
                //printf("El: %u Id: %u\n", in[i].identifier, table[subtable*TABLE_2_SIZE + h2].identifier);
                key_ptr32 joined = {.key = in[i].key, .ptr = table[subtable*TABLE_2_SIZE + h2].key};
                out[out_i] = joined;
                out_i++;
                //hit = 1;
                break;
            }
            //probed[subtable] = table[subtable*TABLE_2_SIZE + h2].key;
        }
        //if (!hit) {
            //printf("%u not found: %u\n", i, in[i].key);
            //printf("El:%u cmp: %u %u %u %u\n", in[i].identifier, probed[0], probed[1],
            //       probed[2], probed[3]);
        //}
    }

    return out_i;
}