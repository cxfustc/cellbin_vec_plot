/***********************************************************
  *File Name: 
  *Description: 
  *Author: Chen Xi
  *Email: chenxi1@genomics.cn
  *Create Time: 2021-03-29 11:37:38
  *Edit History: 
***********************************************************/

#ifndef XDK_IMAGE_H
#define XDK_IMAGE_H

#include <stdint.h>
#include <vector>

/*
 * OpenCV
 */

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h>

#define CV_PALETTE_NCOLORS 29

extern unsigned int xcv_palette_rgb[CV_PALETTE_NCOLORS];
extern float xcv_text_height;
extern double lg_text_scale;

struct xcv_pal_s;
typedef struct xcv_pal_s xcv_pal_t;

struct xcv_pal_s {
	int n, m;
	int * r;
	int * g;
	int * b;
};

#ifdef __cplusplus
extern "C" {
#endif

	int xcv_point_rotate (float row_center, float col_center,
                        float row,        float col,
												float * row_ret,  float * col_ret,
												float angle, int is_clockwise);

	int xcv_center_trans (cv::Mat img, int row_idx, int col_idx);

	int xcv_rigid_rotate (cv::Mat & dst, cv::Mat & src,
			cv::Point2f & center, float alpha, int is_clockwise);

  void xcv_imread (cv::Mat & img, const char * pic);
  void xcv_imread_gray (cv::Mat & img, const char * pic);

  void xcv_crop_3b (cv::Mat & img_orig, cv::Mat & img_new,
      int32_t col_beg, int32_t row_beg, int32_t col_ext, int32_t row_ext);

  void xcv_mean_ub (cv::Mat & img, unsigned char * b, unsigned char * g, unsigned char * r,
      int32_t col_beg, int32_t row_beg, int32_t col_ext, int32_t row_ext);

  void xcv_fill_ub (cv::Mat & img, unsigned char b, unsigned char g, unsigned char r,
      int32_t col_beg, int32_t row_beg, int32_t col_ext, int32_t row_ext);

  void xcv_bgr2rgb_3b (cv::Mat & img);

  void xcv_manual_roi (const char * img_file, std::vector<std::vector<cv::Point>> & contours);

	void xcv_legend (cv::Mat & img, int32_t row_beg, int32_t row_ext,
			             int32_t col_beg, int32_t col_ext,
									 std::vector<cv::Vec3b> & colors, double max_val, const char * tag);

	void xcv_title (cv::Mat & img, int32_t title_row_beg, int32_t title_height, const char * title_content);

	void xcv_test_palette (const char * out_pic, std::vector<cv::Vec3b> & colors);

	void xcv_pie_plot (cv::Point & center);

	void load_colors (const char * palette_file, std::vector<cv::Vec3b> & colors);

	void load_palette_bgr (std::vector<cv::Vec3b> & colors);
	void load_palette_file (std::vector<cv::Vec3b> & colors, const char * file);

	void xcv_scale_bar (cv::Mat & img, int32_t row, int32_t col, int32_t len, const char * tag);

	/*** Palette ***/
	xcv_pal_t * xcv_pal_init (void);
	void xcv_pal_clear (xcv_pal_t * pal);
	void xcv_pal_free (xcv_pal_t * pal);
	void xcv_pal_resize (xcv_pal_t * pal, int size);
	void xcv_pal_add (xcv_pal_t * pal, int r, int g, int b);
	void xcv_create_pal (xcv_pal_t * src_pal, xcv_pal_t * dst_pal, int n_dst_colors);

#ifdef __cplusplus
}
#endif

/*
 * NEF
 */

#endif
