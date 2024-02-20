#ifndef XDK_IMG_H
#define XDK_IMG_H

#include <stdint.h>

/*
 * Basic Image Definitions
 */

struct xrect_s;
typedef struct xrect_s xrect_t;

struct xsize_s;
typedef struct xsize_s xsize_t;

struct ximg_s;
typedef struct ximg_s ximg_t;

struct xrect_s {
  int32_t c, r;
  int32_t w, h;
};

struct xsize_s {
  int32_t w, h;
};

struct ximg_s {
  // dim[0]: n_dims;
  // dim[1]: rows / height
  // dim[2]: cols / width
  // dim[3]: dim3
  int32_t dim[4];

  int64_t n_pixels;
  int64_t m_pixels;
  uint8_t * data;

  int32_t n_channels;
  int32_t depth;
};

#ifdef __cplusplus
extern "C" {
#endif

  ximg_t * ximg_init (void);
  void ximg_clear (ximg_t * img);
  void ximg_free (ximg_t * img);

#ifdef __cplusplus
}
#endif

/*
 * Basic Image Definitions
 */

#include "pair.h"

struct xpoint_s;
typedef struct xpoint_s xpoint_t;

struct xpoint_match_s;
typedef struct xpoint_match_s xpoint_match_t;

struct xpoint_s {
  // world coordinate
  pair_t(dbl) w;

  // local coordinate
  pair_t(dbl) l;
};

struct xpoint_match_s {
  xpoint_t p[2];
  double w;
};

#endif
