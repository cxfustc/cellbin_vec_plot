/***********************************************************
  *File Name: 
  *Description: 
  *Author: Chen Xi
  *Email: chenxi1@genomics.cn
  *Create Time: 2018-11-05 11:22:08
  *Edit History: 
***********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "str.h"
#include "hash.h"
#include "utils.h"
#include "hash_func.h"

/*----------------------------------------------------------------------------*/
/*----------------------------- Shared Functions -----------------------------*/
/*----------------------------------------------------------------------------*/

xh_t *
_xh_init (int64_t size, double load_factor, HashFunc hash_func, IsEqualFunc is_equal_func)
{
  xh_t * hash;

  hash = (xh_t *) ckmalloc (sizeof(xh_t));

  if (hash_func == NULL)
    err_mesg ("hash function is not set!");
  hash->hash_func = hash_func;

  if (is_equal_func == NULL)
    err_mesg ("compare function is not set!");
  hash->is_equal_func = is_equal_func;

  size = size<256 ? 256 : size;
  hash->size = next_prime (size);
  hash->load_factor = (load_factor<=0.0|load_factor>1.0) ? 0.75 : load_factor;
  hash->cnt = hash->del_c = 0;
  hash->max = (uint64_t) (hash->size * hash->load_factor);

  hash->slots = (xh_item_t **) ckalloc (hash->size, sizeof(xh_item_t *));
  hash->pool = (xh_item_t *) ckalloc (hash->max, sizeof(xh_item_t));

  return hash;
}

void
_xh_clear (xh_t * hash)
{
  hash->cnt = hash->del_c = 0;
  memset (hash->slots, 0, hash->size*sizeof(xh_item_t*));
}

void
_xh_free (xh_t * hash)
{
  free (hash->slots);
  free (hash->pool);
  free (hash);
}

/*----------------------------------------------------------------------------*/
/*------------------------------ Set Functions -------------------------------*/
/*----------------------------------------------------------------------------*/

int
_xh_set_add (xh_t * hash, void * key)
{
  uint64_t os;
  uint64_t hash_val;
  xh_item_t * ptr;

  hash_val = hash->hash_func (key);
  os = hash_val % hash->size;
  ptr = hash->slots[os];
  while (ptr) {
    if (hash->is_equal_func(key,ptr->key))
      break;
    ptr = ptr->next;
  }

  if (ptr) {
    if (ptr->deleted) {
      ptr->multi = 1;
      ptr->deleted = 0;
      --hash->del_c;
    } else {
      ++ptr->multi;
    }

    return XH_EXIST;
  }

  if (xh_mem_check(hash, hash->cnt+1) != 0)
    os = hash_val % hash->size;
  ptr = hash->pool + hash->cnt++;
  ptr->key = key;
  ptr->id = hash->cnt - 1;
  ptr->multi = 1;
  ptr->deleted = 0;
  ptr->hash_val = hash_val;
  ptr->next = hash->slots[os];
  hash->slots[os] = ptr;

  return XH_NEW;
}

int
_xh_set_add2 (xh_t * hash, void * new_key, void ** old_key)
{
  uint64_t os;
  uint64_t hash_val;
  xh_item_t * ptr;

  hash_val = hash->hash_func (new_key);
  os = hash_val % hash->size;
  ptr = hash->slots[os];
  while (ptr) {
    if (hash->is_equal_func(new_key,ptr->key))
      break;
    ptr = ptr->next;
  }

  if (ptr) {
    if (ptr->deleted) {
      ptr->multi = 1;
      ptr->deleted = 0;
      --hash->del_c;
    } else {
      ++ptr->multi;
    }
    *old_key = ptr->key;

    return XH_EXIST;
  }

  if (xh_mem_check(hash, hash->cnt+1) != 0)
    os = hash_val % hash->size;
  ptr = hash->pool + hash->cnt++;
  ptr->key = new_key;
  ptr->id = hash->cnt - 1;
  ptr->multi = 1;
  ptr->deleted = 0;
  ptr->hash_val = hash_val;
  ptr->next = hash->slots[os];
  hash->slots[os] = ptr;
  *old_key = NULL;

  return XH_NEW;
}

xh_item_t *
_xh_set_add3 (xh_t * hash, void * key)
{
	uint64_t os;
	uint64_t hash_val;
	xh_item_t * ptr;

	hash_val = hash->hash_func (key);
	os = hash_val % hash->size;
	ptr = hash->slots[os];
	while (ptr) {
		if (hash->is_equal_func(key,ptr->key))
			break;
		ptr = ptr->next;
	}

	if (ptr) {
		if (ptr->deleted) {
			ptr->multi = 1;
			ptr->deleted = 0;
			--hash->del_c;
		} else {
			++ptr->multi;
		}

		return ptr;
	}

	if (xh_mem_check(hash,hash->cnt+1) != 0)
		os = hash_val % hash->size;
	ptr = hash->pool + hash->cnt++;
	ptr->key = key;
	ptr->id = hash->cnt - 1;
	ptr->multi = 1;
	ptr->deleted = 0;
	ptr->hash_val = hash_val;
	ptr->next = hash->slots[os];
	hash->slots[os] = ptr;

	return ptr;
}

int
_xh_set_search (xh_t * hash, void * key)
{
  uint64_t os;
  uint64_t hash_val;
  xh_item_t * ptr;

  hash_val = hash->hash_func (key);
  os = hash_val % hash->size;
  ptr = hash->slots[os];
  while (ptr) {
    if (hash->is_equal_func(key,ptr->key))
      return XH_EXIST;
    ptr = ptr->next;
  }

  return XH_FAIL;
}

void *
_xh_set_search2 (xh_t * hash, void * key)
{
  uint64_t os;
  uint64_t hash_val;
  xh_item_t * ptr;

  hash_val = hash->hash_func (key);
  os = hash_val % hash->size;
  ptr = hash->slots[os];
  while (ptr) {
    if (hash->is_equal_func(key,ptr->key))
      return ptr->key;
    ptr = ptr->next;
  }

  return NULL;
}

xh_item_t *
_xh_set_search3 (xh_t * hash, void * key)
{
  uint64_t os;
  uint64_t hash_val;
  xh_item_t * ptr;

  hash_val = hash->hash_func (key);
  os = hash_val % hash->size;
  ptr = hash->slots[os];
  while (ptr) {
    if (hash->is_equal_func(key,ptr->key))
      return ptr;
    ptr = ptr->next;
  }

  return NULL;
}

/*----------------------------------------------------------------------------*/
/*------------------------------ Map Functions -------------------------------*/
/*----------------------------------------------------------------------------*/

int
_xh_map_add (xh_t * hash, void * key, void * val)
{
  uint64_t os;
  uint64_t hash_val;
  xh_item_t * ptr;

  hash_val = hash->hash_func (key);
  os = hash_val % hash->size;
  ptr = hash->slots[os];
  while (ptr) {
    if (hash->is_equal_func(key,ptr->key))
      break;
    ptr = ptr->next;
  }

  if (ptr) {
    if (ptr->deleted) {
      ptr->multi = 1;
      ptr->deleted = 0;
      --hash->del_c;
    } else {
      ++ptr->multi;
    }
    return XH_EXIST;
  }

  if (xh_mem_check(hash, hash->cnt+1) != 0)
    os = hash_val % hash->size;
  ptr = hash->pool + hash->cnt++;
  ptr->key = key;
  ptr->val = val;
  ptr->id = hash->cnt - 1;
  ptr->multi = 1;
  ptr->deleted = 0;
  ptr->hash_val = hash_val;
  ptr->next = hash->slots[os];
  hash->slots[os] = ptr;

  return XH_NEW;
}

int
_xh_map_add2 (xh_t * hash, void * key, void * val, void ** old_key, void ** old_val)
{
  uint64_t os;
  uint64_t hash_val;
  xh_item_t * ptr;

  hash_val = hash->hash_func (key);
  os = hash_val % hash->size;
  ptr = hash->slots[os];
  while (ptr) {
    if (hash->is_equal_func(key,ptr->key))
      break;
    ptr = ptr->next;
  }

  if (ptr) {
    if (ptr->deleted) {
      ptr->multi = 1;
      ptr->deleted = 0;
      --hash->del_c;
    } else {
      ++ptr->multi;
    }
		*old_key = ptr->key;
		*old_val = ptr->val;

    return XH_EXIST;
  }

  if (xh_mem_check(hash, hash->cnt+1) != 0)
    os = hash_val % hash->size;
  ptr = hash->pool + hash->cnt++;
  ptr->key = key;
  ptr->val = val;
  ptr->id = hash->cnt - 1;
  ptr->multi = 1;
  ptr->deleted = 0;
  ptr->hash_val = hash_val;
  ptr->next = hash->slots[os];
  hash->slots[os] = ptr;

  return XH_NEW;
}

void *
_xh_map_search (xh_t * hash, void * key)
{
  uint64_t os;
  uint64_t hash_val;
  xh_item_t * ptr;

  hash_val = hash->hash_func (key);
  os = hash_val % hash->size;
  ptr = hash->slots[os];
  while (ptr) {
    if (hash->is_equal_func(key,ptr->key))
      return ptr->val;
    ptr = ptr->next;
  }

  return NULL;
}
