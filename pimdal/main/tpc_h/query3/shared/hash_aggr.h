#ifndef _HASHFUNC_H_
#define _HASHFUNC_H_
#include <stdint.h>
#include "datatype.h"

#define AGG_TABLE_SIZE 1024

static uint32_t hash0 (keyptr_out element) {
  uint32_t hash_val = element.orderkey;
  hash_val ^= element.orderdate;
  hash_val ^= element.shippriority;

  return hash_val;
}
#endif