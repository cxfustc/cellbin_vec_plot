/***********************************************************
  *File Name: 
  *Description: 
  *Author: Chen Xi
  *Email: chenxi1@genomics.cn
  *Create Time: 2017-05-13 19:07:11
  *Edit History: 
***********************************************************/

#ifndef XDK_UTILS_H
#define XDK_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <stdint.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/types.h>

#include <zlib.h>

#define XUTILS_STATUS "utested"

/**********************************************************
 ***************** Bio Macro Definitions ******************
 **********************************************************/

#define BASEQ_PHRED_OFFSET 33
#define MAX_READ_BASEQ_SUM 16383

#define int_comp(i) ((i)^0x02)
#define int2base(i) ("ACTGN"[(i)])
#define base2int(b) ((b)>>1&0x03)

extern const char base_rc_tbl[128];
extern const char base2int_tbl[128];
extern const char base2int_tbl_acgt[128];
extern const uint32_t cigar_bwa2samtools[5];

/**********************************************************
 ******************* Macro Definitions ********************
 **********************************************************/

#ifndef XPACKED
#define XPACKED __attribute__ ((__packed__))
#endif

#define IS_N 1
#define NOT_N 0
#define is_N(ch) (((ch)&0xf)==0xe ? IS_N : NOT_N)

#ifndef LINE_MAX
#define LINE_MAX 4096
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#ifndef LONG_LINE_MAX
#define LONG_LINE_MAX 16777216
#endif

#ifndef VLONG_LINE_MAX
#define VLONG_LINE_MAX 1073741824
#endif

/*
 * 1001 9  9
 * 1010 10 a
 * 1011 11 b
 * 1100 12 c
 * 1101 13 d
 * 1110 14 e
 * 1111 15 f
 */

//                      0x0000,0000,000f,ffff
#define U64_KEEP_TAIL20 0x00000000000fffffLLU
//                      0x0000,0000,00ff,ffff
#define U64_KEEP_TAIL24 0x0000000000ffffffLLU

#define ckopen(f,m) (err_ckopen((f),(m), \
			__func__, __FILE__, __LINE__))
#define ckpopen(f,m) (err_ckpopen((f),(m), \
			__func__, __FILE__, __LINE__))
#define ckgzopen(f,m) (err_ckgzopen((f),(m), \
			__func__, __FILE__, __LINE__))
#define ckfwrite(ptr,size,n,fp) (err_ckfwrite((ptr),(size),(n),(fp), \
			__func__, __FILE__, __LINE__))
#define ckfread(ptr,size,n,fp) (err_ckfread((ptr),(size),(n),(fp), \
			__func__, __FILE__, __LINE__))
#define ckfflush(fp) (err_ckfflush((fp), \
			__func__, __FILE__, __LINE__))

#define ckopendir(path) (err_ckopendir((path), \
			__func__, __FILE__, __LINE__))
#define ckcreate_dir(path) (err_ckcreate_dir((path), \
			__func__, __FILE__, __LINE__))

#define rm1file(path) (err_rm1file((path), \
			__func__, __FILE__, __LINE__))
#define rm1dir(path) (err_rm1dir((path), \
			__func__, __FILE__, __LINE__))

#define ckalloc(n,size) (err_ckalloc((n),(size), \
			__func__, __FILE__, __LINE__))
#define ckmalloc(size) (err_ckmalloc((size), \
			__func__, __FILE__, __LINE__))
#define ckrealloc(old,new_size) (err_ckrealloc((old),(new_size), \
			__func__, __FILE__, __LINE__))

#define str_cmp_tail(s1,s2,n) (err_str_cmp_tail((s1),(s2),(n), \
			__func__, __FILE__, __LINE__))
#define str_cut_tail(s,t) (err_str_cut_tail((s),(t), \
			__func__, __FILE__, __LINE__))
#define chomp(s) (err_chomp((s), \
			__func__, __FILE__, __LINE__))
#define chop(s,c) (err_chop((s),(c), \
			__func__, __FILE__, __LINE__))
#define split_string(s,m,c) (err_split_string((s),(m),(c), \
			__func__, __FILE__, __LINE__))
#define is_empty_line(s) (err_is_empty_line((s), \
			__func__, __FILE__, __LINE__))
#define skip_blanks(s) (err_skip_blanks((s), \
      __func__, __FILE__, __LINE__))

#define get_abs_path(p) (err_get_abs_path((p), \
			__func__, __FILE__, __LINE__))
#define get_program_path(p) (err_get_program_path((p), \
			__func__, __FILE__, __LINE__))

#define ALLOC_LINE ((char*)ckalloc(LINE_MAX,sizeof(char)))
#define ALLOC_PATH ((char*)ckalloc(PATH_MAX,sizeof(char)))
#define ALLOC_LONG_LINE ((char*)ckalloc(LONG_LINE_MAX,sizeof(char)))
#define ALLOC_VLONG_LINE ((char*)ckalloc(VLONG_LINE_MAX,sizeof(char)))

#define xswap(a,b,t) ((t)=(a),(a)=(b),(b)=(t))
#define xmax(a,b) ((a)>(b)?(a):(b))
#define xmin(a,b) ((a)<(b)?(a):(b))

#define XDK_TRUE 1
#define XDK_FALSE 0

typedef int (*CompFunc) (const void*, const void*);
typedef int (*IsEqualFunc) (const void*, const void*); // // equal=>1, different=>0

/**********************************************************
 ******************* Inline Functions *********************
 **********************************************************/

static inline int
cmp_itg (const void * a, const void * b)
{
	int32_t * pa = (int32_t *) a;
	int32_t * pb = (int32_t *) b;

	if (*pa < *pb)
		return -1;
	else
		return 1;
}

static inline int
cmp_dbl (const void * a, const void * b)
{
	double * pa = (double *) a;
	double * pb = (double *) b;

	if (*pa < *pb)
		return -1;
	else
		return 1;
}

/**********************************************************
 *************** IO, FileSystem Functions *****************
 **********************************************************/

#ifdef __cplusplus
extern "C" {
#endif

	FILE * err_ckopen (const char * file, const char * mode,
			const char * c_func, const char * c_file, int line_c);

	FILE * err_ckpopen (const char * cmd, const char * mode,
			const char * c_func, const char * c_file, int line_c);

	gzFile err_ckgzopen (const char * file, const char * mode,
			const char * c_func, const char * c_file, int line_c);

	size_t err_ckfwrite (const void * ptr, size_t size, size_t n_elem, FILE * fp,
			const char * c_func, const char * c_file, int line_c);
	size_t err_ckfread (void * ptr, size_t size, size_t n_elem, FILE * fp,
			const char * c_func, const char * c_file, int line_c);

	int err_ckfflush (FILE * fp,
			const char * c_func, const char * c_file, int line_c);

	DIR * err_ckopendir (const char * path,
			const char * c_func, const char * c_file, int line_c);

	int err_ckcreate_dir (const char * path,
			const char * c_func, const char * c_file, int line_c);

	int err_rm1file (const char * path,
			const char * c_func, const char * c_file, int line_c);

	int err_rm1dir (const char * path,
			const char * c_func, const char * c_file, int line_c);

	void * err_ckalloc (size_t n_elem, size_t size_elem,
			const char * c_func, const char * c_file, int line_c);

	void * err_ckmalloc (size_t size,
			const char * c_func, const char * c_file, int line_c);

	void * err_ckrealloc (void * old, size_t new_size,
			const char * c_func, const char * c_file, int line_c);

#ifdef __cplusplus
}
#endif

/**********************************************************
 ********************* Log Functions **********************
 **********************************************************/

#ifdef __cplusplus
extern "C" {
#endif

	void err_mesg (const char * format, ...);

	void warn_mesg (const char * format, ...);

#ifdef __cplusplus
}
#endif

/**********************************************************
 ************* String Manipulation Functions **************
 **********************************************************/

#ifdef __cplusplus
extern "C" {
#endif

	int err_str_cmp_tail (const char * s1, const char * s2, int len,
			const char * c_func, const char * c_file, int line_c);

	int err_str_cut_tail (char * str, const char * tail,
			const char * c_func, const char * c_file, int line_c);

	int err_chomp (char * str,
			const char * c_func, const char * c_file, int line_c);

	int err_chop (char * str, char c,
			const char * c_func, const char * c_file, int line_c);

	char ** err_split_string (char * string, char * markers, int * item_c,
			const char * c_func, const char * c_file, int line_c);

	int err_is_empty_line (char * line,
			const char * c_func, const char * c_file, int line_c);

  char * err_skip_blanks (char * line,
      const char * c_func, const char * c_file, int line_c);

#ifdef __cplusplus
}
#endif

/**********************************************************
 ******************** Misc Functions **********************
 **********************************************************/

#ifdef __cplusplus
extern "C" {
#endif

	char * err_get_abs_path (const char * path,
			const char * c_func, const char * c_file, int line_c);

	char * err_get_program_path (const char * program,
			const char * c_func, const char * c_file, int line_c);

	int cksystem (const char * cmd);

	uint64_t next_prime (uint64_t num);

	int int2deci_nbits (int num);

	void time_diff (struct timeval * beg, struct timeval * end, struct timeval * diff);

	void time_div (struct timeval * all, struct timeval * unit, int n_part);

  int ckpthread_create (pthread_t * pid, const pthread_attr_t * attr,
      void* (*start_routine)(void*), void * arg);
  int ckpthread_join (pthread_t pid);
  void send_work_signal (int * signals, int work_signal, int wait_signal, int n_thread);

	void time_segm_accu (struct timeval * total, struct timeval * cur_seg_beg);
	double timeval2sec (struct timeval * tv);

	int time_parse (char * str, int32_t * h, int32_t * m, int32_t * s);

  void file2new_dir (char * new_path, const char * new_dir, const char * out_path);

  // need free
  char * get_necessary_param (int argc, char ** argv, int opt_idx);

#ifdef __cplusplus
}
#endif

static inline int
xround (double d)
{
  return (int) (d + 0.5f);
}

/******************/
/****** ACTG ******/
/******************/

static inline uint64_t
enc64 (char * s, int l)
{
  int i;
  uint64_t b;
  uint64_t cs;

  if (l > 20)
    err_mesg ("Only support sequences no longer than 20 nt!");

  cs = 0;
  for (i=0; i<l; ++i) {
    b = (uint64_t) (s[i] & 0x7);
    cs = (cs << 3) | b;
  }

  return cs;
}

#define idx_dec(i) ("NANCTNNG"[(i)])

static inline void
dec64 (FILE * out, uint64_t cs, int l)
{
	int i;
	int of;

	for (i=l-1; i>=0; --i) {
		of = i * 3;
		fprintf (out, "%c", idx_dec((cs>>of)&0x7));
	}
}

static inline uint32_t
enc32 (char * s, int l)
{
  int i;
  uint32_t b;
  uint32_t cs;

  if (l > 10)
    err_mesg ("Only support sequences no longer than 20 nt!");

  cs = 0;
  for (i=0; i<l; ++i) {
    b = (uint32_t) (s[i] & 0x7);
    cs = (cs << 3) | b;
  }

  return cs;
}

static inline void
dec32 (FILE * out, uint32_t cs, int l)
{
	int i;
	int of;

	for (i=l-1; i>=0; --i) {
		of = i * 3;
		fprintf (out, "%c", idx_dec((cs>>of)&0x7));
	}
}

/******************/
/****** ACGT ******/
/******************/

static inline uint64_t
enc64_acgt (char * s, int l)
{
  int i;
  uint64_t b;
  uint64_t cs;

  if (l > 20)
    err_mesg ("Only support sequences no longer than 20 nt!");

  cs = 0;
  for (i=0; i<l; ++i) {
    b = (uint64_t) (base2int_tbl_acgt[s[i]]);
    cs = (cs << 3) | b;
  }

  return cs;
}

#define idx_dec_acgt(i) ("ACGTNNNN"[(i)])

static inline void
dec64_acgt (FILE * out, uint64_t cs, int l)
{
	int i;
	int of;

	for (i=l-1; i>=0; --i) {
		of = i * 3;
		fprintf (out, "%c", idx_dec_acgt((cs>>of)&0x7));
	}
}

static inline uint32_t
enc32_acgt (char * s, int l)
{
  int i;
  uint32_t b;
  uint32_t cs;

  if (l > 10)
    err_mesg ("Only support sequences no longer than 20 nt!");

  cs = 0;
  for (i=0; i<l; ++i) {
    b = (uint64_t) (base2int_tbl_acgt[s[i]]);
    cs = (cs << 3) | b;
  }

  return cs;
}

static inline void
dec32_acgt (FILE * out, uint32_t cs, int l)
{
	int i;
	int of;

	for (i=l-1; i>=0; --i) {
		of = i * 3;
		fprintf (out, "%c", idx_dec_acgt((cs>>of)&0x7));
	}
}

/**********************************************************
 ********************* Option Parse ***********************
 **********************************************************/

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif
