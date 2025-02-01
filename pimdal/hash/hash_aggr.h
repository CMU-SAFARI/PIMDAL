#ifndef _HASHAGGR_H_
#define _HASHAGGR_H_

#include <stdint.h>
#include "datatype.h"

static uint32_t hash0 (key_ptr_t element) {
    uint32_t key = element.key;
    key += 272333;
    key += ~(key << 5);
    key ^= (key >> 18);
    key += (key << 11);
    key ^= (key >> 17);
    //key += ~(key << 11);
    //key ^= (key >> 16);
    return key;
}
#endif