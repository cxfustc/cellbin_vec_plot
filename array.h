/*************************************************
 * File Name: ../../ds/array.h
 * Description: 
 * Author: Chen Xi
 * Mail: chenxi1@genomics.cn
 * Created Time: 2018/11/30
 * Edit History: 
 *************************************************/

#ifndef XDK_ARRAY_H
#define XDK_ARRAY_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "utils.h"

#define ARR_INIT_SIZE 2

#define ARR_SORTED   1
#define ARR_UNIQSORT 2

#define ARR_DEF(name, type_t) \
	struct arr_##name##_s { \
		type_t * arr; \
		int64_t n, m; \
    int32_t * multi; \
    int32_t multi_max; \
    uint32_t flag; \
	}; \
	typedef struct arr_##name##_s arr_##name##_t; \
	\
	static inline arr_##name##_t * arr_init_##name (void) { \
		arr_##name##_t * arr; \
		arr = (arr_##name##_t *) ckalloc (1, sizeof(arr_##name##_t)); \
		arr->n = 0; \
		arr->m = ARR_INIT_SIZE; \
		arr->arr = (type_t*) ckalloc (ARR_INIT_SIZE, sizeof(type_t)); \
    arr->multi_max = -1; \
		return arr; \
	} \
	\
	static inline void arr_free_##name ( \
			arr_##name##_t * arr) { \
		free (arr->arr); \
    if (arr->multi_max > 0) \
      free (arr->multi); \
		free (arr); \
	} \
	\
	static inline void arr_clear_##name ( \
			arr_##name##_t * arr) { \
		arr->n = 0; \
		memset (arr->arr, 0, arr->m*sizeof(type_t)); \
	} \
	\
	static inline void arr_resize_##name ( \
			arr_##name##_t * arr, \
			int64_t new_size) { \
		int64_t old_max; \
		if (new_size <= arr->m) \
			return; \
		old_max = arr->m; \
		while (new_size > arr->m) { \
			if (arr->m < 0x10000) \
				arr->m <<= 1; \
			else \
				arr->m += 0x10000; \
		} \
		arr->arr = (type_t*) ckrealloc (arr->arr, arr->m*sizeof(type_t)); \
		memset (arr->arr+old_max, 0, (arr->m-old_max)*sizeof(type_t)); \
	} \
	\
	static inline type_t arr_at_##name ( \
			arr_##name##_t * arr, \
			int64_t idx) { \
    if (idx >= arr->n) \
      err_mesg ("[arr_at] offset exceeds limit!"); \
		return arr->arr[idx]; \
	} \
  \
  static inline void arr_add_##name ( \
      arr_##name##_t * arr, \
      type_t val) { \
    arr_resize_##name (arr, arr->n+1); \
    arr->arr[arr->n++] = val; \
    arr->flag = 0; \
  } \
  \
  static inline void arr_update_##name ( \
      arr_##name##_t * arr, \
      int64_t idx, \
      type_t val) { \
    if (idx >= arr->n) \
      err_mesg ("[arr_at] offset exceeds limit!"); \
    arr->arr[idx] = val; \
    arr->flag = 0; \
  } \
  \
  static inline void arr_insert_##name ( \
      arr_##name##_t * arr, \
      type_t val, \
      int64_t idx) { \
    if (idx < 0) \
      err_mesg ("[arr_insert] idx must >= 0!"); \
    arr_resize_##name (arr, idx+1); \
    arr->n = idx + 1; \
    arr->arr[idx] = val; \
    arr->flag = 0; \
  } \
  \
  static inline void arr_copy_##name ( \
      arr_##name##_t * dst, \
      arr_##name##_t * src) { \
    arr_resize_##name (dst, src->n); \
    dst->n = src->n; \
    memcpy (dst->arr, src->arr, src->n*sizeof(type_t)); \
    dst->flag = 0; \
  } \
  \
  static inline void arr_append_##name ( \
      arr_##name##_t * dst, \
      arr_##name##_t * src) { \
    arr_resize_##name (dst, dst->n+src->n); \
    memcpy (dst->arr+dst->n, src->arr, src->n*sizeof(type_t)); \
    dst->n += src->n; \
    dst->flag = 0; \
  } \
  \
  static type_t arr_last_##name ( \
      arr_##name##_t * arr) { \
    if (arr->n <= 0) \
      err_mesg ("array is empty!"); \
    return arr->arr[arr->n-1]; \
  } \
  \
	static int __array_##name##_def_end__ = 1

#define arr_t(name) arr_##name##_t
#define arr_cnt(arr) ((arr)->n)

#define arr_init(name) arr_init_##name()
#define arr_free(name,arr) arr_free_##name(arr)
#define arr_clear(name,arr) arr_clear_##name(arr)
#define arr_resize(name,arr,new_size) arr_resize_##name((arr),(new_size))
#define arr_at(name,arr,idx) arr_at_##name((arr),(idx))
#define arr_add(name,arr,val) arr_add_##name((arr),(val))
#define arr_update(name,arr,idx,val) arr_update_##name((arr),(idx),(val))
#define arr_copy(name,dst,src) arr_copy_##name((dst),(src))
#define arr_append(name,dst,src) arr_append_##name((dst),(src))
#define arr_last(name,arr) arr_last_##name(arr)

ARR_DEF (i8,  int8_t);
ARR_DEF (u8,  uint8_t);
ARR_DEF (i32, int32_t);
ARR_DEF (u32, uint32_t);
ARR_DEF (i64, int64_t);
ARR_DEF (u64, uint64_t);
ARR_DEF (flt, float);
ARR_DEF (dbl, double);
ARR_DEF (dpt, double*);
ARR_DEF (ldl, long double);
ARR_DEF (cpt, char*);

#define ARR_ABS(x) ((x)<0 ? (-(x)) : (x))

#define ARR_ABS_DEF(name,type_t) \
	static inline type_t arr_abs_max_##name ( \
			arr_##name##_t * arr) { \
		int64_t i; \
		type_t max, abs_val; \
		assert (arr->n >= 1); \
		max = ARR_ABS (arr->arr[0]); \
		for (i=1; i<arr->n; ++i) { \
			abs_val = ARR_ABS (arr->arr[i]); \
			if (abs_val > max) \
				max = abs_val; \
		} \
		return max; \
	} \
	static int __array_abs_##name##_def_end__ = 1

#define arr_abs_max(name,arr) arr_abs_max_##name(arr)

ARR_ABS_DEF (dbl, double);
ARR_ABS_DEF (ldl, double);

#define ARR_NUM_DEF(name,type_t) \
	static inline type_t arr_max_##name ( \
			arr_##name##_t * arr) { \
		int64_t i; \
		type_t max; \
		assert (arr->n >= 1); \
		max = arr->arr[0]; \
		for (i=1; i<arr->n; ++i) \
			if (arr->arr[i] > max) \
				max = arr->arr[i]; \
		return max; \
	} \
  \
  static inline int64_t arr_imin2_##name ( \
      arr_##name##_t * arr, \
      int64_t beg, \
      int64_t end) { \
    int64_t i, min_idx; \
    type_t min; \
    assert (beg >= 0); \
    assert (beg < end); \
    assert (end <= arr->n); \
    min = arr->arr[beg]; \
    min_idx = beg; \
    for (i=beg+1; i<end; ++i) \
      if (arr->arr[i] < min) { \
        min = arr->arr[i]; \
        min_idx = i; \
      } \
    return min_idx; \
  } \
  \
  static inline type_t arr_min2_##name ( \
      arr_##name##_t * arr, \
      int64_t beg, \
      int64_t end) { \
    int64_t i; \
    type_t min; \
    assert (beg >= 0); \
    assert (beg < end); \
    assert (end <= arr->n); \
    min = arr->arr[beg]; \
    for (i=beg+1; i<end; ++i) \
      if (arr->arr[i] < min) \
        min = arr->arr[i]; \
    return min; \
  } \
  \
  static inline int64_t arr_find_##name ( \
      arr_##name##_t * arr, \
      type_t target) { \
    int64_t i; \
    for (i=0; i<arr->n; ++i) \
      if (arr->arr[i] == target) \
        break; \
    if (i == arr->n) \
      return -1;\
    else \
      return i; \
  } \
  \
  static inline int64_t arr_find_between_##name ( \
      arr_##name##_t * arr, \
      type_t target, \
      int64_t beg, \
      int64_t end) { \
    assert (beg >= 0); \
    assert (beg < end); \
    assert (end <= arr->n); \
    int64_t i; \
    for (i=beg; i<end; ++i) \
      if (arr->arr[i] == target) \
        break; \
    if (i == end) \
      return -1; \
    else \
      return i; \
  } \
	static int __array_num_##name##_def_end__ = 1

#define ARR_SORT_DEF(name,type_t) \
  static inline int arr_comp_func_##name ( \
      const void * a, \
      const void * b) { \
    type_t * pa = (type_t *) a; \
    type_t * pb = (type_t *) b; \
    if (*pa > *pb) \
      return 1; \
    else if (*pa < *pb) \
      return -1; \
    else \
      return 0; \
    /*return memcmp (a, b, sizeof(type_t));*/ \
  } \
  \
  static inline void arr_std_sort_##name ( \
      arr_##name##_t * arr) { \
    qsort (arr->arr, arr->n, sizeof(type_t), arr_comp_func_##name); \
    arr->flag |= ARR_SORTED; \
  } \
  \
  static inline void arr_multi_mem_resize_##name ( \
      arr_##name##_t * arr) { \
    assert (arr->n >= 0); \
    if (arr->multi_max < 0) { \
      arr->multi_max = ((arr->n>>3) + 1) << 3; \
      arr->multi = (int32_t *) ckalloc (arr->multi_max, sizeof(int32_t)); \
    } else if (arr->multi_max < arr->n) { \
      arr->multi_max = ((arr->n>>3) + 1) << 3; \
      arr->multi = (int32_t *) realloc (arr->multi, arr->multi_max*sizeof(int32_t)); \
    } \
  } \
  \
  static inline void arr_uniqsort_##name ( \
      arr_##name##_t * arr) { \
    if (arr->n <= 1) { \
      arr_multi_mem_resize_##name (arr); \
      arr->multi[0] = 1; \
      return; \
    } \
    int32_t i, j, k; \
    type_t tmp; \
    type_t* dst; \
    type_t* src; \
    qsort (arr->arr, arr->n, sizeof(type_t), arr_comp_func_##name); \
    arr_multi_mem_resize_##name (arr); \
    i = k = 0; \
    dst = arr->arr + i; \
    for (j=1; j<arr->n; ++j) { \
      src = arr->arr + j; \
      if (*dst != *src) { \
        arr->multi[i]=j-k; k=j; \
        ++i, ++dst; \
        if (i != j) { \
          tmp = *dst; \
          *dst = *src; \
          *src = tmp; \
        } \
      } \
    } \
    arr->multi[i] = j-k; \
    arr->n = i + 1; \
    arr->flag |= ARR_SORTED; \
    arr->flag |= ARR_UNIQSORT; \
  } \
  \
  static double arr_percentile_##name ( \
      arr_##name##_t * arr, \
      int percentile) { \
    if (percentile<0 || percentile>100) \
      err_mesg ("percentile must >=0 and <= 100!"); \
    if (!(arr->flag & ARR_SORTED)) \
      err_mesg ("array must be sorted first!"); \
    if (percentile == 0) \
      return (double)arr->arr[0]; \
    if (percentile == 100) \
      return (double)arr->arr[arr->n-1]; \
    double q = percentile / 100.0 * (arr->n-1); \
    int cq = (int) floor (q); \
    double vi = arr->arr[cq+1] - arr->arr[cq]; \
    return arr->arr[cq] + (q-cq)*vi; \
  } \
  \
	static int __array_sort_##name##_def_end__ = 1

#define arr_max(name,arr) arr_max_##name(arr)
#define arr_imin2(name,arr,beg,end) arr_imin2_##name((arr),(beg),(end))
#define arr_min2(name,arr,beg,end) arr_min2_##name((arr),(beg),(end))
#define arr_find(name,arr,target) arr_find_##name((arr),(target))
#define arr_find_between(name,arr,target,beg,end) arr_find_between_##name((arr),(target),(beg),(end))
#define arr_std_sort(name,arr) arr_std_sort_##name(arr)
#define arr_uniqsort(name,arr) arr_uniqsort_##name(arr)
#define arr_percentile(name,arr,q) arr_percentile_##name((arr),(q))

ARR_NUM_DEF (i8,  int8_t);
ARR_NUM_DEF (u8,  uint8_t);

ARR_NUM_DEF (i32, int32_t);
ARR_SORT_DEF (i32, int32_t);
ARR_SORT_DEF (flt, float);
ARR_SORT_DEF (dbl, double);

ARR_NUM_DEF (u32, uint32_t);
ARR_NUM_DEF (i64, int64_t);
ARR_NUM_DEF (u64, uint64_t);
ARR_NUM_DEF (dbl, double);
ARR_NUM_DEF (ldl, long double);

#endif
