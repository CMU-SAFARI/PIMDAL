#ifndef _HASHFUNC_H_
#define _HASHFUNC_H_
#include <stdint.h>
#include "datatype.h"

#ifdef UNIQUE
#define AGG_TABLE_SIZE 4096

static uint32_t hash0 (key_ptr32 element) {
    uint32_t key = element.ptr;
    key += 272333;
    key += ~(key << 5);
    key ^= (key >> 18);
    key += (key << 11);
    key ^= (key >> 17);
    //key += ~(key << 11);
    //key ^= (key >> 16);
    return key;
}

inline bool empty(key_ptr32 in) {
    return in.key == 0xffffffff;
}

inline bool duplicate(key_ptr32 element, key_ptr32 curr) {
    bool res = (curr.ptr == element.ptr);
    return res;
}
#else
#define AGG_TABLE_SIZE 1024

static uint32_t hash0 (key_ptrtext element) {
    uint32_t key = 0;
    for (uint32_t i = 0; i < 7; i++) {
        key += element.prio[2*i];
        key ^= element.prio[2*i + 1];
    }

    return key;
}

inline bool empty(key_ptrtext in) {
    return in.prio[0] == '\xFF';
}

inline bool duplicate(key_ptrtext element, key_ptrtext curr) {
    bool res = (strcmp(element.prio, curr.prio) == 0);
    return res;
}
#endif

#endif