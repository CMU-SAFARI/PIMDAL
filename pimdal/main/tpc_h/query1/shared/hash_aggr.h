#ifndef _HASHFUNC_H_
#define _HASHFUNC_H_
#include <stdint.h>
#include "datatype.h"

#define AGG_TABLE_SIZE 256

static uint32_t hash0 (key_ptrout element) {

    return element.l_linestatus ^ element.l_returnflag;
}

inline bool empty(key_ptrout in) {
    return (in.l_returnflag == '\xFF') &&
           (in.l_linestatus == '\xFF');
}

inline bool duplicate(key_ptrout element, key_ptrout curr) {
    bool res = (element.l_returnflag == curr.l_returnflag) &&
               (element.l_linestatus == curr.l_linestatus);
    return res;
}

#endif
