#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <stdbool.h>

#define AGG_TABLE_SIZE 4096
#define AGG_BLOCK_BYTES 512

typedef struct
{
    uint32_t key;
    uint32_t ptr;
} key_ptr32;

typedef struct
{
    int64_t key;
    uint32_t ptr;
    uint32_t pad;
} key_ptr64;

typedef struct
{
    char str[28];
} text;


typedef struct
{
    char key[28];
    uint32_t ptr;
} key_ptrtext;

// Our main pointer datatype
typedef key_ptr32 key_ptr_t;

inline bool empty(key_ptr_t in) {
    return in.key == 0xffffffff;
}

inline bool duplicate(key_ptr_t element, key_ptr_t curr) {
    return element.key == curr.key;
}

#ifndef ENERGY
#define ENERGY 0
#endif
#define PRINT 0

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define divceil(n, m) (((n)-1) / (m) + 1)
#define roundup(n, m) ((n / m) * m + m)
#endif