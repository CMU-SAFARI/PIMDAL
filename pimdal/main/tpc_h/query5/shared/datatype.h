#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <string.h>

#define SEL_BLOCK_BYTES 1024
#define AGG_BLOCK_BYTES 1024
#ifndef PTR_TYPE
#define PTR_TYPE key_ptr32
#endif

typedef struct
{
    uint32_t key;
    uint32_t ptr;
} key_ptr32;

typedef struct
{
    char key[32];
    uint32_t ptr;
    uint32_t pad;
} key_ptrtext;

typedef struct
{
    uint32_t key;
    uint32_t pad;
    int64_t revenue;
} keyval_out;

// Our main pointer datatype
typedef PTR_TYPE key_ptr_t;

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