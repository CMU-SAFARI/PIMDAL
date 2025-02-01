#ifndef _HASHFUNC_H_
#define _HASHFUNC_H_
#include <stdint.h>

static uint32_t hash0 (char key[15]) {
  uint32_t hash_val = 0;
  for (uint32_t i = 0; i < 15; i++) {
    hash_val ^= key[i];
  }

  return hash_val;
}
#endif