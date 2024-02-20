/***********************************************************
  *File Name: 
  *Description: 
  *Author: Chen Xi
  *Email: chenxi1@genomics.cn
  *Create Time: 2021-04-08 15:47:47
  *Edit History: 
***********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "img.h"
#include "utils.h"

ximg_t *
ximg_init (void)
{
  ximg_t * img;

  img = (ximg_t *) ckalloc (1, sizeof(ximg_t));

  return img;
}

void
ximg_clear (ximg_t * img)
{
  img->n_pixels = 0;
  img->n_channels = 0;
  img->depth = 0;
}

void
ximg_free (ximg_t * img)
{
  if (img->data)
    free (img->data);
  free (img);
}
