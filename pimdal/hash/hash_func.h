#ifndef _HASHFUNC_H_
#define _HASHFUNC_H_
#include <stdint.h>

static uint32_t hash0 (uint32_t key) {
    key += 272333;
    key += ~(key << 5);
    key ^= (key >> 18);
    key += (key << 11);
    key ^= (key >> 17);
    //key += ~(key << 11);
    //key ^= (key >> 16);
    return key;
}

static uint32_t hash1 (uint32_t key) {
    key += 593627;
    key += ~(key << 22);
    key ^= (key >> 26);
    key += (key << 10);
    key ^= (key >> 28);
    //key += ~(key << 12);
    //key ^= (key >> 13);
  return key;
}

static uint32_t hash2 (uint32_t key) {
    key += 274277;
    key += ~(key << 13);
    key ^= (key >> 20);
    key += (key << 3);
    key ^= (key >> 9);
    //key += ~(key << 8);
    //key ^= (key >> 10);
  return key;
}

static uint32_t hash3 (uint32_t key) {
    key += 855467;
    key += ~(key << 4);
    key ^= (key >> 21);
    key += (key << 5);
    key ^= (key >> 16);
    //key += ~(key << 8);
    //key ^= (key >> 12);
  return key;
}

static uint32_t hash4 (uint32_t key) {
    key += 176521;
    key += ~(key << 10);
    key ^= (key >> 25);
    key += (key << 5);
    key ^= (key >> 24);
    //key += ~(key << 15);
    //key ^= (key >> 13);
  return key;
}

uint32_t (*hash_func[4])(uint32_t key) = {hash1, hash2, hash3, hash4};
#endif