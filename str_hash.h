#ifndef XDK_STR_HASH_H
#define XDK_STR_HASH_H

#include "str.h"
#include "hash.h"
#include "hash_func.h"

static inline uint64_t
str_hash_func (const void * key)
{
  str_t * s = (str_t *) key;
  return blizzard_hash_func (s->s, s->l, 1);
}

HASH_SET_DEF (xstr, str_t);
#define str_hash_t xh_set_t(xstr)
#define str_hash_cnt(h) xh_set_cnt(h)

#define str_hash_init() \
	xh_set_init(xstr,4096,0.75,str_init2,str_copy,str_hash_func,str_equal2)

#define str_hash_init2(size) \
  xh_set_init(xstr,(size),0.75,str_init2,str_copy,str_hash_func,str_equal2)

#define str_hash_free(h) \
	xh_set_free(xstr,(h),str_free2)

#define str_hash_clear(h) \
	xh_set_clear(xstr,(h),str_clear)

#define str_hash_add(h,key) \
	xh_set_add(xstr,(h),(key))

#define str_hash_add2(h,key) \
	xh_set_add2(xstr,(h),(key))

#define str_hash_add3(h,key) \
	xh_set_add3(xstr,(h),(key))

#define str_hash_search(h,key) \
	xh_set_search(xstr,(h),(key))

#define str_hash_search2(h,key) \
	xh_set_search2(xstr,(h),(key))

#define str_hash_search3(h,key) \
	xh_set_search3(xstr,(h),(key))

#define str_hash_merge(dst,src) \
  xh_set_merge(xstr,(dst),(src))

HASH_MAP_DEF (s2i, str_t, int32_t);
#define str2int_map_t xh_map_t(s2i)
#define str2int_map_init() \
  xh_map_init(s2i,4096,0.75,str_init2,NULL,str_copy,NULL,str_hash_func,str_equal2)

HASH_MAP_DEF (s2s, str_t, str_t);
#define str2str_map_t xh_map_t(s2s)
#define str2str_map_init() \
  xh_map_init(s2s,4096,0.75,str_init2,str_init2,str_copy,str_copy,str_hash_func,str_equal2)

#define str2str_map_init2(size) \
  xh_map_init(s2s,(size),0.75,str_init2,str_init2,str_copy,str_copy,str_hash_func,str_equal2)

#endif
