/***********************************************************
  *File Name: 
  *Description: 
  *Author: Chen Xi
  *Email: chenxi1@genomics.cn
  *Create Time: 2018-11-02 16:16:44
  *Edit History: 
***********************************************************/

#ifndef _XDK_HASH_H
#define _XDK_HASH_H

/*
 * Warning: These hash functions are not thread safe
 */

#include <assert.h>
#include <stdint.h>

#include "bmp.h"
#include "utils.h"

#define XH_FAIL 0
#define XH_EXIST 1
#define XH_NEW 2

typedef uint64_t (*HashFunc) (const void*);

struct xh_item_s;
typedef struct xh_item_s xh_item_t;

struct xh_s;
typedef struct xh_s xh_t;

struct xh_item_s {
  void * key, * val;
  uint64_t hash_val;
  int64_t id;
  uint32_t multi:31;
  uint32_t deleted:1;
  xh_item_t * next;
};

struct xh_s {
  uint64_t size;
  uint64_t max;
  uint64_t cnt;
  uint64_t del_c;
  double load_factor;
  HashFunc hash_func;
  IsEqualFunc is_equal_func;
  xh_item_t * pool;
  xh_item_t ** slots;
};

/*---------------------------------------------------------------------------*/
/*---------------------------- Static Functions -----------------------------*/
/*---------------------------------------------------------------------------*/

static inline int
xh_mem_check (xh_t * hash, uint64_t new_cnt)
{
	uint64_t i;
	uint64_t os;
	uint64_t old_max;
	uint64_t new_size;
	xh_item_t * ptr;

	if (new_cnt <= hash->max)
		return 0;

	old_max = hash->max;
	new_size = hash->size;
	do {
		if (new_size < 0xFFFFFFFU)
			new_size <<= 1;
		else
			new_size += 0xFFFFFFFU;
		new_size = next_prime (new_size);
	} while ((uint64_t)((double)new_size*hash->load_factor) < new_cnt);

	hash->size = new_size;
	hash->max = (uint64_t) ((double)new_size * hash->load_factor);

	hash->pool = (xh_item_t *) ckrealloc (hash->pool, hash->max*sizeof(xh_item_t));

	free (hash->slots);
	hash->slots = (xh_item_t **) ckalloc (hash->size, sizeof(xh_item_t*));
	for (i=0; i<hash->cnt; i++) {
		ptr = hash->pool + i;
		os = ptr->hash_val % hash->size;

		ptr->next = hash->slots[os];
		hash->slots[os] = ptr;
	}

	return 1;
}

/**********************************************************
 ********************* Shared Functions *******************
 **********************************************************/

#ifdef __cplusplus
extern "C" {
#endif

xh_t * _xh_init (int64_t size, double load_factor, HashFunc hash_f, IsEqualFunc is_equal_f);
void _xh_clear (xh_t * h);
void _xh_free (xh_t * h);

#ifdef __cplusplus
}
#endif

/**********************************************************
 ************************ Hash Set ************************
 **********************************************************/

#ifdef __cplusplus
extern "C" {
#endif

/*
 * if key is new added to set:
 *   return XH_NEW
 * if key already exist in set:
 *   return XH_EXIST
 * if key fail to add
 *   return XH_FAIL
 */
int _xh_set_add (xh_t * h, void * key);

/*
 * if key is new added to set:
 *   *old_key==NULL and return XH_NEW
 * if key already exist in set:
 *   *old_key==old_key_in_hash and return XH_EXIST
 * if key fail to add
 *   return XH_FAIL
 */
int _xh_set_add2 (xh_t * h, void * new_key, void ** old_key);

/*
 * normally, it will return the xh_item_t
 * if encounter errors, it will return NULL
 */
xh_item_t * _xh_set_add3 (xh_t * h, void * key);

/*
 * if key is in the set:
 *   return XH_EXIST
 * if key is not found in the set:
 *   return XH_FAIL
 */
int _xh_set_search (xh_t * h, void * key);

/*
 * if key is in the set:
 *   return the existed original key
 * if key is not found in the set:
 *   return NULL
 */
void * _xh_set_search2 (xh_t * h, void * key);

/*
 * if key is in the set:
 *   return the existed original xh_item_t
 * if key is not found in the set:
 *   return NULL
 */
xh_item_t * _xh_set_search3 (xh_t * h, void * key);

#ifdef __cplusplus
}
#endif

#define HASH_SET_DEF(name, key_t) \
  BMP_DEF (xh_set_key##name, key_t); \
  struct xh_set_##name##_s { \
    xh_t * hash; \
    bmp_t(xh_set_key##name) * key_bmp; \
    void (*key_copy_f) (key_t*, key_t*); \
  }; \
  typedef struct xh_set_##name##_s xh_set_##name##_t; \
  \
  static inline void xh_set_init2_##name ( \
			xh_set_##name##_t * xh, \
      int64_t size, double load_factor, \
      void (*key_init_f)(key_t*), \
      void (*key_copy_f)(key_t*,key_t*), \
      HashFunc hash_f, \
      IsEqualFunc is_equal_f) { \
    xh->hash = _xh_init (size, load_factor, hash_f, is_equal_f); \
    xh->key_bmp = bmp_init (xh_set_key##name, size, key_init_f); \
    xh->key_copy_f = key_copy_f; \
  } \
	\
  static inline xh_set_##name##_t * xh_set_init_##name ( \
      int64_t size, double load_factor, \
      void (*key_init_f)(key_t*), \
      void (*key_copy_f)(key_t*,key_t*), \
      HashFunc hash_f, \
      IsEqualFunc is_equal_f) { \
    xh_set_##name##_t * xh; \
    xh = (xh_set_##name##_t *) ckmalloc (sizeof(xh_set_##name##_t)); \
    xh->hash = _xh_init (size, load_factor, hash_f, is_equal_f); \
    xh->key_bmp = bmp_init (xh_set_key##name, size, key_init_f); \
    xh->key_copy_f = key_copy_f; \
    return xh; \
  } \
  \
  static inline void xh_set_clear_##name ( \
      xh_set_##name##_t * xh, \
      void (*key_clear_f) (key_t*)) { \
    bmp_clear (xh_set_key##name, xh->key_bmp, key_clear_f); \
    _xh_clear (xh->hash); \
  } \
  \
  static inline void xh_set_free_##name ( \
      xh_set_##name##_t * xh, \
      void (*key_free_f) (key_t*)) { \
    bmp_free (xh_set_key##name, xh->key_bmp, key_free_f); \
    _xh_free (xh->hash); \
    free (xh); \
  } \
  \
  static inline int xh_set_add_##name ( \
      xh_set_##name##_t * xh, \
      key_t * key) { \
    int ret; \
    key_t * new_key; \
    new_key = bmp_alloc (xh_set_key##name, xh->key_bmp); \
    if (xh->key_copy_f == NULL) \
      memcpy (new_key, key, sizeof(key_t)); \
    else \
      xh->key_copy_f (new_key, key); \
    ret = _xh_set_add (xh->hash, (void*)new_key); \
    if (ret != XH_NEW) \
      bmp_pop (xh_set_key##name, xh->key_bmp); \
    return ret; \
  } \
  static inline key_t * xh_set_add2_##name ( \
      xh_set_##name##_t * xh, \
      key_t * key) { \
    int ret; \
    key_t * new_key; \
    key_t * old_key; \
    new_key = bmp_alloc (xh_set_key##name, xh->key_bmp); \
    if (xh->key_copy_f == NULL) \
      memcpy (new_key, key, sizeof(key_t)); \
    else \
      xh->key_copy_f (new_key, key); \
    ret = _xh_set_add2 (xh->hash, (void*)new_key, (void**)(&old_key)); \
    if (ret != XH_NEW) { \
      bmp_pop (xh_set_key##name, xh->key_bmp); \
      new_key = old_key; \
    } \
    return new_key; \
  } \
  \
  static inline xh_item_t * xh_set_add3_##name ( \
      xh_set_##name##_t * xh, \
      key_t * key) { \
		int64_t old_cnt; \
		xh_item_t * ptr; \
		key_t * new_key; \
		new_key = bmp_alloc (xh_set_key##name, xh->key_bmp); \
		if (xh->key_copy_f == NULL) \
			memcpy (new_key, key, sizeof(key_t)); \
		else \
			xh->key_copy_f (new_key, key); \
		old_cnt = xh->hash->cnt; \
		ptr = _xh_set_add3 (xh->hash, (void*)new_key); \
		if (xh->hash->cnt == old_cnt) \
			bmp_pop (xh_set_key##name, xh->key_bmp); \
		else \
			assert (xh->hash->cnt == old_cnt+1); \
		return ptr; \
	} \
	\
  static inline int xh_set_search_##name ( \
      xh_set_##name##_t * xh, \
      key_t * key) { \
    return _xh_set_search (xh->hash, (void*)key); \
  } \
  static inline key_t * xh_set_search2_##name ( \
      xh_set_##name##_t * xh, \
      key_t * key) { \
    return (key_t*) _xh_set_search2 (xh->hash, (void*)key); \
  } \
  static inline xh_item_t * xh_set_search3_##name ( \
      xh_set_##name##_t * xh, \
      key_t * key) { \
    return _xh_set_search3 (xh->hash, (void*)key); \
  } \
  static inline int xh_set_key_iter_init_##name ( \
      xh_set_##name##_t * xh) { \
    return bmp_iter_init_xh_set_key##name (xh->key_bmp); \
  } \
  static inline key_t * xh_set_key_iter_next_##name ( \
      xh_set_##name##_t * xh) { \
    return bmp_iter_next_xh_set_key##name (xh->key_bmp); \
  } \
  \
  static inline void xh_set_merge_##name ( \
      xh_set_##name##_t * dst, \
      xh_set_##name##_t * src) { \
    uint64_t i; \
    uint64_t os; \
    key_t * new_key; \
    xh_t * dhash = dst->hash; \
    xh_t * shash = src->hash; \
    xh_item_t * ditem; \
    xh_item_t * sitem; \
    xh_item_t * ptr; \
    for (i=0; i<shash->cnt; ++i) { \
      sitem = shash->pool + i; \
      os = sitem->hash_val % dhash->size; \
      ptr = dhash->slots[os]; \
      while (ptr) { \
        if (dhash->is_equal_func(sitem->key,ptr->key)) \
          break; \
        ptr = ptr->next; \
      } \
      if (ptr) { \
        if (ptr->deleted) { \
          ptr->multi = sitem->multi; \
          ptr->deleted = 0; \
          --dhash->del_c; \
        } else {\
          ptr->multi += sitem->multi; \
        } \
        continue; \
      } \
      new_key = bmp_alloc (xh_set_key##name, dst->key_bmp); \
      if (dst->key_copy_f == NULL) \
        memcpy (new_key, sitem->key, sizeof(key_t)); \
      else \
        dst->key_copy_f (new_key, (key_t*)sitem->key); \
      if (xh_mem_check(dhash,dhash->cnt+1) != 0) \
        os = sitem->hash_val % dhash->size; \
      ptr = dhash->pool + dhash->cnt++; \
      ptr->key = new_key; \
      ptr->id = dhash->cnt - 1; \
      ptr->multi = sitem->multi; \
      ptr->deleted = 0; \
      ptr->hash_val = sitem->hash_val; \
      ptr->next = dhash->slots[os]; \
      dhash->slots[os] = ptr; \
    } \
  } \
  \
  static int __xh_set_##name##_def_end__ = 1

#define xh_set_t(name) xh_set_##name##_t
#define xh_set_cnt(xh) ((xh)->hash->cnt)

#define xh_set_init2(name, xh, size, load_factor, key_init_f, key_copy_f, key_hash_f, key_equal_f) \
  xh_set_init2_##name ((xh), (size), (load_factor), (key_init_f), (key_copy_f), (key_hash_f), (key_equal_f))

#define xh_set_init(name, size, load_factor, key_init_f, key_copy_f, key_hash_f, key_equal_f) \
  xh_set_init_##name ((size), (load_factor), (key_init_f), (key_copy_f), (key_hash_f), (key_equal_f))

#define xh_set_clear(name, xh, key_clear_f) \
  xh_set_clear_##name ((xh), (key_clear_f))

#define xh_set_free(name, xh, key_free_f) \
  xh_set_free_##name ((xh), (key_free_f))

#define xh_set_add(name, xh, key) \
  xh_set_add_##name ((xh), (key))

#define xh_set_add2(name, xh, key) \
  xh_set_add2_##name ((xh), (key))

#define xh_set_add3(name, xh, key) \
  xh_set_add3_##name ((xh), (key))

#define xh_set_search(name, xh, key) \
  xh_set_search_##name ((xh), (key))

#define xh_set_search2(name, xh, key) \
  xh_set_search2_##name ((xh), (key))

#define xh_set_search3(name, xh, key) \
  xh_set_search3_##name ((xh), (key))

#define xh_set_key_iter_init(name, xh) \
  xh_set_key_iter_init_##name (xh)

#define xh_set_key_iter_next(name, xh) \
  xh_set_key_iter_next_##name (xh)

#define xh_set_merge(name, dst, src) \
  xh_set_merge_##name ((dst), (src))

static inline uint64_t
i32_hash_f (const void * key)
{
  return (uint64_t) (*(int32_t*)key);
}
static inline int
i32_equal_f (const void * a, const void * b)
{
  int32_t va = * (int32_t *) a;
  int32_t vb = * (int32_t *) b;

  return va == vb;
}
HASH_SET_DEF (i32, int32_t);
#define i32_hash_t xh_set_t(i32)

static inline i32_hash_t *
xh_i32_set_init (void)
{
	return xh_set_init (i32,16,0.75,NULL,NULL,i32_hash_f,i32_equal_f);
}

static inline void
xh_i32_set_init2 (i32_hash_t * hash)
{
	xh_set_init2 (i32,hash,16,0.75,NULL,NULL,i32_hash_f,i32_equal_f);
}

//#define xh_i32_set_init() xh_set_init(i32,16,0.75,NULL,NULL,i32_hash_f,i32_equal_f)

static inline uint64_t
u32_hash_f (const void * key)
{
  return (uint64_t) (*(uint32_t*)key);
}
static inline int
u32_equal_f (const void * a, const void * b)
{
  uint32_t va = * (uint32_t *) a;
  uint32_t vb = * (uint32_t *) b;

  return va == vb;
}
HASH_SET_DEF (u32, uint32_t);
#define u32_hash_t xh_set_t(u32)

static inline u32_hash_t *
xh_u32_set_init (void)
{
	return xh_set_init (u32,16,0.75,NULL,NULL,u32_hash_f,u32_equal_f);
}

static inline void
xh_u32_set_init2 (u32_hash_t * hash)
{
	xh_set_init2 (u32,hash,16,0.75,NULL,NULL,u32_hash_f,u32_equal_f);
}

//#define xh_u32_set_init() xh_set_init(u32,16,0.75,NULL,NULL,u32_hash_f,u32_equal_f)

static inline uint64_t
u64_hash_f (const void * key)
{
  return *(uint64_t*)key;
}
static inline int
u64_equal_f (const void * a, const void * b)
{
  uint64_t va = * (uint64_t *) a;
  uint64_t vb = * (uint64_t *) b;

  return va == vb;
}
HASH_SET_DEF (u64, uint64_t);
#define xh_u64_set_init() xh_set_init(u64,16,0.75,NULL,NULL,u64_hash_f,u64_equal_f)

/**********************************************************
 ************************ Hash Maps ***********************
 **********************************************************/

#ifdef __cplusplus
extern "C" {
#endif

/*
 * if key is new added to set:
 *   return XH_NEW
 * if key already exist in set:
 *   return XH_EXIST
 * if key fail to add
 *   return XH_FAIL
 */
int _xh_map_add (xh_t * h, void * key, void * val);

int _xh_map_add2 (xh_t * h, void * key, void * val, void ** updated_key, void ** updated_val);

/*
 * if key is in the set:
 *   return val
 * if key is not found in the set:
 *   return NULL
 */
void * _xh_map_search (xh_t * h, void * key);

#ifdef __cplusplus
}
#endif

#define HASH_MAP_DEF(name, key_t, val_t) \
  BMP_DEF (xh_map_key##name, key_t); \
  BMP_DEF (xh_map_val##name, val_t); \
  struct xh_map_##name##_s { \
    xh_t * hash; \
    bmp_t(xh_map_key##name) * key_bmp; \
    bmp_t(xh_map_val##name) * val_bmp; \
    void (*key_copy_f) (key_t*,key_t*); \
    void (*val_copy_f) (val_t*,val_t*); \
  }; \
  typedef struct xh_map_##name##_s xh_map_##name##_t; \
  \
  static inline xh_map_##name##_t * xh_map_init_##name ( \
      int64_t size, double load_factor, \
      void (*key_init_f)(key_t*), \
      void (*val_init_f)(val_t*), \
      void (*key_copy_f)(key_t*,key_t*), \
      void (*val_copy_f)(val_t*,val_t*), \
      HashFunc hash_f, \
      IsEqualFunc is_equal_f) { \
    xh_map_##name##_t * xh; \
    xh = (xh_map_##name##_t *) ckmalloc (sizeof(xh_map_##name##_t)); \
    xh->hash = _xh_init (size, load_factor, hash_f, is_equal_f); \
    xh->key_bmp = bmp_init (xh_map_key##name, size, key_init_f); \
    xh->val_bmp = bmp_init (xh_map_val##name, size, val_init_f); \
    xh->key_copy_f = key_copy_f; \
    xh->val_copy_f = val_copy_f; \
    return xh; \
  } \
  \
  static inline void xh_map_clear_##name ( \
      xh_map_##name##_t * xh, \
      void (*key_clear_f) (key_t*), \
      void (*val_clear_f) (val_t*)) { \
    bmp_clear (xh_map_key##name, xh->key_bmp, key_clear_f); \
    bmp_clear (xh_map_val##name, xh->val_bmp, val_clear_f); \
    _xh_clear (xh->hash); \
  } \
  \
  static inline void xh_map_free_##name ( \
      xh_map_##name##_t * xh, \
      void (*key_free_f) (key_t*), \
      void (*val_free_f) (val_t*)) { \
    bmp_free (xh_map_key##name, xh->key_bmp, key_free_f); \
    bmp_free (xh_map_val##name, xh->val_bmp, val_free_f); \
    _xh_free (xh->hash); \
    free (xh); \
  } \
  \
  static inline int xh_map_add_##name ( \
      xh_map_##name##_t * xh, \
      key_t * key, \
      val_t * val) { \
    int ret; \
    key_t * new_key; \
    val_t * new_val; \
    new_key = bmp_alloc (xh_map_key##name, xh->key_bmp); \
    if (xh->key_copy_f == NULL) \
      memcpy (new_key, key, sizeof(key_t)); \
    else \
      xh->key_copy_f (new_key, key); \
    new_val = bmp_alloc (xh_map_val##name, xh->val_bmp); \
    if (xh->val_copy_f == NULL) \
      memcpy (new_val, val, sizeof(val_t)); \
    else \
      xh->val_copy_f (new_val, val); \
    ret = _xh_map_add (xh->hash, (void*)new_key, (void*)new_val); \
    if (ret != XH_NEW) { \
      bmp_pop (xh_map_key##name, xh->key_bmp); \
      bmp_pop (xh_map_val##name, xh->val_bmp); \
    } \
    return ret; \
  } \
  \
  static inline int xh_map_add2_##name ( \
      xh_map_##name##_t * xh, \
      key_t * key, \
      val_t * val, \
			key_t ** updated_key, \
			val_t ** updated_val) { \
    int ret; \
    key_t * new_key; \
    val_t * new_val; \
    new_key = bmp_alloc (xh_map_key##name, xh->key_bmp); \
    if (xh->key_copy_f == NULL) \
      memcpy (new_key, key, sizeof(key_t)); \
    else \
      xh->key_copy_f (new_key, key); \
    new_val = bmp_alloc (xh_map_val##name, xh->val_bmp); \
    if (xh->val_copy_f == NULL) \
      memcpy (new_val, val, sizeof(val_t)); \
    else \
      xh->val_copy_f (new_val, val); \
    ret = _xh_map_add2 (xh->hash, (void*)new_key, (void*)new_val, \
				(void**)updated_key, (void**)updated_val); \
    if (ret != XH_NEW) { \
      bmp_pop (xh_map_key##name, xh->key_bmp); \
      bmp_pop (xh_map_val##name, xh->val_bmp); \
    } else { \
			assert (ret == XH_NEW); \
			*updated_key = new_key; \
			*updated_val = new_val; \
		} \
    return ret; \
  } \
	\
  static inline val_t * xh_map_search_##name ( \
      xh_map_##name##_t * xh, \
      key_t * key) { \
    return (val_t*) _xh_map_search (xh->hash, key); \
  } \
  \
  static inline int xh_map_key_iter_init_##name ( \
      xh_map_##name##_t * xh) { \
    return bmp_iter_init_xh_map_key##name (xh->key_bmp); \
  } \
  \
  static inline key_t * xh_map_key_iter_next_##name ( \
      xh_map_##name##_t * xh) { \
    return bmp_iter_next_xh_map_key##name (xh->key_bmp); \
  } \
	\
	static inline int xh_map_val_iter_init_##name ( \
			xh_map_##name##_t * xh) { \
		return bmp_iter_init_xh_map_val##name (xh->val_bmp); \
	} \
  \
	static inline val_t * xh_map_val_iter_next_##name ( \
			xh_map_##name##_t * xh) { \
		return bmp_iter_next_xh_map_val##name (xh->val_bmp); \
	} \
  \
  static inline void xh_map_copy_##name ( \
      xh_map_##name##_t * dst, \
      xh_map_##name##_t * src) { \
    int ret; \
    key_t * dst_key; \
    key_t * src_key; \
    val_t * dst_val; \
    val_t * src_val; \
    if (dst->hash->cnt != 0) \
      err_mesg ("destination map need to be cleared first!"); \
    assert (src->key_bmp->n_item == src->val_bmp->n_item); \
    if (bmp_iter_init_xh_map_key##name(src->key_bmp) != 0) \
      err_mesg ("fail to init key interation for srouce map!"); \
    if (bmp_iter_init_xh_map_val##name(src->val_bmp) != 0) \
      err_mesg ("fail to init key interation for srouce map!"); \
    while ((src_key = bmp_iter_next_xh_map_key##name(src->key_bmp)) != NULL) { \
      assert ((src_val = bmp_iter_next_xh_map_val##name(src->val_bmp)) != NULL); \
      dst_key = bmp_alloc (xh_map_key##name, dst->key_bmp); \
      if (dst->key_copy_f == NULL) \
        memcpy (dst_key, src_key, sizeof(key_t)); \
      else \
        dst->key_copy_f (dst_key, src_key); \
      dst_val = bmp_alloc (xh_map_val##name, dst->val_bmp); \
      if (dst->val_copy_f == NULL) \
        memcpy (dst_val, src_val, sizeof(val_t)); \
      else \
        dst->val_copy_f (dst_val, src_val); \
      ret = _xh_map_add (dst->hash, (void*)dst_key, (void*)dst_val); \
      assert (ret == XH_NEW); \
    } \
  } \
  \
  static inline void xh_map_iter_init_##name ( \
      xh_map_##name##_t * xh) { \
    assert (xh->key_bmp->n_item == xh->val_bmp->n_item); \
    assert (bmp_iter_init_xh_map_key##name(xh->key_bmp) == 0); \
    assert (bmp_iter_init_xh_map_val##name(xh->val_bmp) == 0); \
  } \
  \
  static inline int xh_map_iter_next_##name ( \
      xh_map_##name##_t * xh, \
      key_t ** key_pp, val_t ** val_pp) { \
    if ((*key_pp = bmp_iter_next_xh_map_key##name(xh->key_bmp)) == NULL) \
      return -1; \
    if ((*val_pp = bmp_iter_next_xh_map_val##name(xh->val_bmp)) == NULL) \
      return -1; \
    return 0; \
  } \
  \
  static inline void xh_map_dump_##name ( \
      xh_map_##name##_t * xh, \
      FILE * out, \
      void (*key_dump_f) (FILE*,key_t*), \
      void (*val_dump_f) (FILE*,val_t*)) { \
    key_t * key; \
    val_t * val; \
    xh_map_iter_init_##name (xh); \
    while (xh_map_iter_next_##name(xh,&key,&val) == 0) { \
      if (key_dump_f != NULL) \
        key_dump_f (out, key); \
      if (val_dump_f != NULL) \
        val_dump_f (out, val); \
      fprintf (out, "\n"); \
    } \
  } \
  \
  static int __xh_map_##name##_def_end__ = 1

#define xh_map_t(name) xh_map_##name##_t
#define xh_map_cnt(xh) ((xh)->hash->cnt)

#define xh_map_init(name, size, load_factor, key_init_f, val_init_f, key_copy_f, val_copy_f, key_hash_f, key_equal_f) \
  xh_map_init_##name (size, load_factor, key_init_f, val_init_f, key_copy_f, val_copy_f, key_hash_f, key_equal_f)

#define xh_map_clear(name, xh, key_clear_f, val_clear_f) \
  xh_map_clear_##name (xh, key_clear_f, val_clear_f)

#define xh_map_free(name, xh, key_free_f, val_free_f) \
  xh_map_free_##name (xh, key_free_f, val_free_f)

#define xh_map_add(name, xh, key, val) \
  xh_map_add_##name (xh, key, val)

#define xh_map_add2(name, xh, key, val, updated_key, updated_val) \
  xh_map_add2_##name ((xh), (key), (val), (updated_key), (updated_val))

#define xh_map_search(name, xh, key) \
  xh_map_search_##name (xh, key)

#define xh_map_key_iter_init(name,xh) \
  xh_map_key_iter_init_##name (xh)

#define xh_map_key_iter_next(name,xh) \
  xh_map_key_iter_next_##name (xh)

#define xh_map_val_iter_init(name,xh) \
  xh_map_val_iter_init_##name (xh)

#define xh_map_val_iter_next(name,xh) \
  xh_map_val_iter_next_##name (xh)

#define xh_map_copy(name,dst,src) \
  xh_map_copy_##name ((dst), (src))

#define xh_map_iter_init(name,xh) \
  xh_map_iter_init_##name (xh)

#define xh_map_iter_next(name,xh,kpp,vpp) \
  xh_map_iter_next_##name ((xh),(kpp),(vpp))

#define xh_map_dump(name,xh,fp,key_dump_f,val_dump_f) \
  xh_map_dump_##name ((xh), (fp), key_dump_f,val_dump_f)

HASH_MAP_DEF (i2i, int32_t, int32_t);

HASH_MAP_DEF (u64_i32, uint64_t, int32_t);
#define xh_u64_i32_map_init() xh_map_init(u64_i32,16,0.75,NULL,NULL,NULL,NULL,u64_hash_f,u64_equal_f)

HASH_MAP_DEF (u32_u32, uint32_t, uint32_t);
#define xh_u32_u32_map_init() xh_map_init(u32_u32,16,0.75,NULL,NULL,NULL,NULL,u32_hash_f,u32_equal_f)

HASH_MAP_DEF (u64_f64, uint64_t, double);
#define xh_u64_f64_map_init() xh_map_init(u64_f64,16,0.75,NULL,NULL,NULL,NULL,u64_hash_f,u64_equal_f)
#define xh_u64_f64_map_clear(xh) xh_map_clear(u64_f64,(xh),NULL,NULL)

#endif
