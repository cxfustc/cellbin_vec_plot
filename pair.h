#ifndef XDK_PAIR_H
#define XDK_PAIR_H

#include <stdint.h>

#define PAIR_DEF(name, type_t) \
  struct pair_##name##_s { \
    type_t x, y; \
  }; \
  typedef struct pair_##name##_s pair_##name##_t; \
  \
  static int __pair_##name##_def_end__ = 1

#define pair_t(name) pair_##name##_t

PAIR_DEF (itg, int);
PAIR_DEF (flt, float);
PAIR_DEF (dbl, double);

PAIR_DEF (i32, int32_t);

#endif
