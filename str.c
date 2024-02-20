/***********************************************************
  *File Name: 
  *Description: 
  *Author: Chen Xi
  *Email: chenxi1@genomics.cn
  *Create Time: 2017-05-16 16:45:00
  *Edit History: 
***********************************************************/

#include <stdio.h>
#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "str.h"
#include "utils.h"

#define STR_INIT_SIZE 8

str_t *
str_init (void)
{
	str_t * s;

	s = (str_t *) ckalloc (1, sizeof(str_t));
	str_init2 (s);

	return s;
}

void
str_init2 (str_t * s)
{
	s->l = 0;
	s->m = STR_INIT_SIZE;
	s->s = (char *) ckalloc (s->m, sizeof(char));
}

void
str_free (str_t * s)
{
	if (s == NULL)
		err_mesg ("[%s] s==NULL!", __func__);
	free (s->s);
	free (s);
}

void
str_free2 (str_t * s)
{
	if (s == NULL)
		err_mesg ("[%s] s==NULL!", __func__);
	free (s->s);
}

void
str_clear (str_t * s)
{
	if (s == NULL)
		err_mesg ("[%s] s==NULL!", __func__);
	s->l = 0;
  s->s[0] = 0;
}

int
str_resize (str_t * s, int64_t l)
{
	if (s == NULL)
		err_mesg ("[%s] s==NULL!", __func__);

	if (s->m > l)
		return 0;

	if (s->m == 0)
		s->m = STR_INIT_SIZE;
	while (s->m <= l) {
    if (s->m < 0x10000)
  		s->m <<= 1;
    else
      s->m += 0x10000;
  }
	s->s = (char *) ckrealloc (s->s, s->m*sizeof(char));

	return 0;
}

int
str_dump (FILE * fp, str_t * s)
{
	s->s[s->l+1] = '\0';
	fprintf (fp, "%s\n", s->s);
	return 0;
}

int
str_write (FILE * fp, str_t * s)
{
	fwrite (&s->l, 4, 1, fp);
	fwrite (s->s, 1, s->l+1, fp);

	return 0;
}

int
str_read (FILE * fp, str_t * s)
{
  int ret;

	ret = fread (&s->l, 4, 1, fp);
	str_resize (s, s->l);
	ret = fread (s->s, 1, s->l+1, fp);

	return 0;
}

void
str_copy (str_t * dst, str_t * src)
{
	if (dst == NULL)
		err_mesg ("[%s] dst==NULL!", __func__);
	if (src == NULL)
		err_mesg ("[%s] src==NULL!", __func__);

	str_resize (dst, src->l);
	memcpy (dst->s, src->s, src->l);
	dst->l = src->l;
	dst->s[dst->l] = '\0';
}

str_t *
str_dup (str_t * str)
{
	str_t * dst;

	if (str == NULL)
		err_mesg ("[%s] str == NULL!", __func__);

	dst = str_init ();
	str_copy (dst, str);

	return dst;
}

int
str_assign (str_t * str, const char * s)
{
	int64_t l;

	l = strlen (s);
	str_resize (str, l);
	str->l = l;
	memcpy (str->s, s, l+1);

	return 0;
}

int
str_assign2 (str_t * str, const char * s, int64_t l)
{
  if (l <= 0)
    return -1;

  str_resize (str, l);
  str->l = l;
  if (l > 0)
    memcpy (str->s, s, l);
  str->s[l] = '\0';

  return 0;
}

int
str_assign3 (str_t * str, const char * s, char stop_tag)
{
	const char * ch;
	int64_t l;

	for (l=0,ch=s; (*ch!=stop_tag)&&(*ch!='\0'); ++ch,++l)
		;

	str_resize (str, l);
	if (l > 0)
		memcpy (str->s, s, l);
	str->s[l] = '\0';

	return 0;
}

int
str_append (str_t * str, const char * s, int64_t l)
{
  if (l <= 0)
    return -1;

	str_resize (str, str->l+l);
	memcpy (str->s+str->l, s, l);
	str->l += l;
  str->s[str->l] = '\0';

	return 0;
}

int
str_add (str_t * str, char ch)
{
  str_resize (str, str->l+1);
  str->s[str->l++] = ch;
	str->s[str->l] = '\0';

  return 0;
}

void
str_pop (str_t * str)
{
  str->s[str->l-1] = '\0';
  --(str->l);
}

void
str_chop (str_t * str, char ch)
{
  if (str->s[str->l-1 == ch]) {
    str->s[str->l-1] = '\0';
    --(str->l);
  }
}

int
str_cmp (str_t * s1, str_t * s2)
{
	int ret;

  if (s1->l == 0) {
    if (s2->l == 0)
      return 0;
    else
      return -1;
  }

  if (s2->l == 0)
    return 1;

	if (s1->l < s2->l) {
		if ((ret=memcmp(s1->s,s2->s,s1->l)) == 0)
			return -1;
		else
			return ret;
	} else if (s1->l > s2->l) {
		if ((ret=memcmp(s1->s,s2->s,s2->l)) == 0)
			return 1;
		else
			return ret;
	} else
		return memcmp (s1->s, s2->s, s1->l);
}

int
str_cmp2 (const void * s1, const void * s2)
{
  return str_cmp((str_t*)s1, (str_t*)(s2));
}

int
str_ncmp (str_t * s1, const char * s2, int64_t l)
{
  int ret;

  if (s1->l < l) {
    if ((ret=memcmp(s1->s,s2,s1->l)) == 0)
      return -1;
    else
      return ret;
  } else
    return memcmp (s1->s, s2, l);
}

int
str_equal (str_t * s1, str_t * s2)
{
	int64_t i;

	if (s1->l != s2->l)
		return XDK_FALSE;

	for (i=0; i<s1->l; ++i)
		if (s1->s[i] != s2->s[i])
			return XDK_FALSE;

	return XDK_TRUE;
}

int
str_equal2 (const void * s1, const void * s2)
{
  return str_equal ((str_t*)s1, (str_t*)s2);
}

int
str_nequal (str_t * s1, const char * s2, int64_t l)
{
  int64_t i;

  if (s1->l != l)
    return XDK_FALSE;

  for (i=0; i<l; ++i)
    if (s1->s[i] != s2[i])
      return XDK_FALSE;

  return XDK_TRUE;
}

int
str_is_empty (str_t * s)
{
  return s->l == 0;
}

void
str_update (str_t * s)
{
  s->l = strlen (s->s);
}

void
str_shrink (str_t * s)
{
  char * c1;
  char * c2;
  int i;

  for (c1=s->s+s->l-1,i=s->l; i>0; --c1,--i)
    if (!isspace(*c1))
      break;
  s->l = i;
  s->s[s->l] = '\0';

  for (c1=s->s,i=0; i<s->l; ++i,++c1)
    if (!isspace(s->s[i]))
      break;
  if (i == 0)
    return;

  for (c2=s->s; i<=s->l; ++i,++c1,++c2)
    *c2 = *c1;
}

void
str_set_add (str_set_t * set, const char * seq)
{
  str_t * s = str_set_alloc (set);
  str_assign (s, seq);
}

void
str_set_add2 (str_set_t * set, const char * seq, int l)
{
  str_t * s = str_set_alloc (set);
  str_assign2 (s, seq, l);
}

void
str_set_copy (str_set_t * dst, str_set_t * src)
{
	int64_t i;
	str_t * d, * s;

	if (dst==NULL || src==NULL)
		err_mesg ("dst or src is NULL!");

	mp_resize (xstr, dst, mp_cnt(src));
  dst->n = src->n;
	for (i=0; i<mp_cnt(src); i++) {
		d = mp_at (xstr, dst, i);
		s = mp_at (xstr, src, i);
		str_copy (d, s);
	}
}

str_set_t *
str_load_list (const char * file_list)
{
  char * line;
  char * file;
  FILE * fp;
  str_t * s;
  str_set_t * files;

  files = str_set_init ();

  line = ALLOC_LINE;
  file = ALLOC_LINE;
  fp = ckopen (file_list, "r");
  while (fgets(line, LINE_MAX, fp)) {
    if (is_empty_line(line))
      continue;
    chomp (line);
    sscanf (line, "%s", file);
    s = str_set_alloc (files);
    str_assign (s, file);
  }
  fclose (fp);
  free (line);
  free (file);

  return files;
}

void
str_set_dump (const char * file, str_set_t * files)
{
  int i;
  FILE * out;
  str_t * s;

  if (*file=='-')
    out = stdout;
  else
    out = ckopen (file, "w");

  for (i=0; i<str_set_cnt(files); ++i) {
    s = str_set_at (files, i);
    fprintf (out, "%s\n", s->s);
  }
  //printf ("\ntotal %d files\n", i);

  if (*file != '-')
    fclose (out);
}

str_tok_t *
str_token_init (char * flag_str)
{
  char * ch;
  str_tok_t * tok;

  tok = (str_tok_t *) ckmalloc (sizeof(str_tok_t));
  tok->flags = (char *) ckalloc (128, 1);
  tok->segms = str_set_init ();

  for (ch=flag_str; *ch!='\0'; ++ch)
    tok->flags[*ch] = 1;

  return tok;
}

void
str_token_reset (str_tok_t * tok, char * flag_str)
{
  char * ch;

  memset (tok->flags, 0, 128);
  for (ch=flag_str; *ch!='0'; ++ch)
    tok->flags[*ch] = 1;
}

void
str_token_split (str_t * info, str_tok_t * tok_set)
{
  char * start;
  char * end;
  char * flags;
  int i;
  str_set_t * segms;

  flags = tok_set->flags;
  segms = tok_set->segms;

  str_set_clear (segms);
  start = info->s;
  for (end=info->s,i=0; i<info->l; ++end,++i) {
    if (flags[*end]) {
      str_set_add2 (segms, start, end-start);
      start = end + 1;
    }
  }
  str_set_add2 (segms, start, end-start);
}

#define X_TOK_EMPTY 0
#define X_TOK_VALUE 1

void
str_token_split2 (str_t * info, str_tok_t * tok_set)
{
  char * start;
  char * end;
  char * flags;
  int i;
  int pre_st;
  int cur_st;
  str_set_t * segms;

  flags = tok_set->flags;
  segms = tok_set->segms;

  str_set_clear (segms);
  pre_st = X_TOK_EMPTY;
  for (end=info->s,i=0; i<info->l; ++end,++i) {
    if (flags[*end])
      cur_st = X_TOK_EMPTY;
    else
      cur_st = X_TOK_VALUE;

    if (pre_st==X_TOK_EMPTY && cur_st==X_TOK_VALUE) {
      start = end;
    } else if (pre_st==X_TOK_VALUE && cur_st==X_TOK_EMPTY) {
      str_set_add2 (segms, start, end-start);
    }

    pre_st = cur_st;
  }
  if (pre_st == X_TOK_VALUE)
    str_set_add2 (segms, start, end-start);
}
