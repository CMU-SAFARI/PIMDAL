#ifndef _HASHAGGR_H_
#define _HASHAGGR_H_

#include <stdint.h>
#include "datatype.h"

#define AGG_TABLE_SIZE 2048

static uint32_t hash0 (keyval_out element) {
    uint32_t key = element.key;
    key += 272333;
    key += ~(key << 5);
    key ^= (key >> 18);
    key += (key << 11);
    key ^= (key >> 17);
    return key;
}

inline bool empty(keyval_out in) {
    return in.key == 0xffffffff;
}

inline bool duplicate(keyval_out element, keyval_out curr) {
    bool res = (element.key == curr.key);
    return res;
}

#endif