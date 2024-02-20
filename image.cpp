/***********************************************************
  *File Name: 
  *Description: 
  *Author: Chen Xi
  *Email: chenxi1@genomics.cn
  *Create Time: 2021-03-29 12:46:40
  *Edit History: 
***********************************************************/

#include <stdio.h>
#include <wchar.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "image.h"
#include "utils.h"

/*
 * OpenCV
 */

using namespace cv;
using namespace std;

double lg_text_scale = 1.0;

typedef struct {
	int u; // upper border
	int d; // down border
	int l; // left border
	int r; // right_border
} xcv_border_t;

unsigned int xcv_palette_rgb[] = {
                                  0xE41A1C, 0x377EB8, 0x4DAF4A, 0x984EA3, 0xFF7F00,
                                  0xA65628, 0xFFFF33, 0xF781BF, 0x999999, 0xE5D8BD,
                                  0xB3CDE3, 0xCCEBC5, 0xFED9A6, 0xFBB4AE, 0x8DD3C7,
                                  0xBEBADA, 0x80B1D3, 0xB3DE69, 0xFCCDE5, 0xBC80BD,
                                  0xFFED6F, 0x8DA0CB, 0xE78AC3, 0xE5C494, 0xFB9A99,
                                  0xE31A1C, 0xCAB2D6, 0x6A3D9A, 0xB15928};

float xcv_text_height = 40.0;

/*
 * clockwise
 */

int
xcv_point_rotate (float cx, float cy, // center for rotation
                  float x,  float y,
									float*xr, float*yr,
									float angle, int is_clockwise)
{
	float fx = x - cx;
	float fy = y - cy;
	float x1, y1;
	float sign, r;

	sign = is_clockwise ? (1.0) : (-1.0);

	r = angle / 180 * 3.14;
	x1 = fx*cos(r) + sign*fy*sin(r);
	y1 = -sign*fx*sin(r) + fy*cos(r);

	*xr = x1 + cx;
	*yr = y1 + cy;

	return 0;
}

int
xcv_center_trans (cv::Mat img, int row_idx, int col_idx)
{
	return 0;
}

static void
border_check (xcv_border_t * b, float row, float col, int row_max, int col_max)
{
	int val;

	if (row < 0) {
		val = (int) ceil (-row);
		if (val > b->u)
			b->u = val;
	} else {
		val = (int) ceil (row);
		val -= row_max;
		if (val > b->d)
			b->d = val;
	}

	if (col < 0) {
		val = (int) ceil (-col);
		if (val > b->l)
			b->l = val;
	} else {
		val = (int) ceil (col);
		val -= col_max;
		if (val > b->r)
			b->r = val;
	}
}

static void
determine_border_color (Mat src, int * b, int * g, int * r)
{
	int i, j, k;
	long val[3];
	long cnt;

	val[0] = val[1] = val[2] = cnt = 0;
	for (i=k=0; i<src.rows&&k<5; ++i,++k) {
		for (j=0; j<src.cols; ++j) {
			val[0] += src.at<Vec3b>(i,j)[0];
			val[1] += src.at<Vec3b>(i,j)[1];
			val[2] += src.at<Vec3b>(i,j)[2];
			++cnt;
		}
	}

	*b = val[0] / cnt;
	*g = val[1] / cnt;
	*r = val[2] / cnt;
}

// center: (col.idx, row.idx)
int
xcv_rigid_rotate (Mat & dst, Mat & src, Point2f & center, float angle, int is_clockwise)
{
	char file[4096];
	int i;
	int b, g, r;
	int min_dist;
	float min_distf;
	float rad;
	float row, col;
	float row1, col1;
	Point2f pt_src[4];
	Point2f pt_dst[4];
	xcv_border_t bdr;
	Mat ext_src;

	// extend borders
	bdr.u = bdr.d = bdr.l = bdr.r = 0;
	// upper-left
	row=0, col=0;
	xcv_point_rotate (center.y, center.x, row, col, &row1, &col1, angle, is_clockwise);
	pt_src[0] = Point2f (col, row);
	pt_dst[0] = Point2f (col1, row1);
	border_check (&bdr, row1, col1, src.rows, src.cols);

	// upper-right
	row=0, col=src.cols;
	xcv_point_rotate (center.y, center.x, row, col, &row1, &col1, angle, is_clockwise);
	pt_src[1] = Point2f (col, row);
	pt_dst[1] = Point2f (col1, row1);
	border_check (&bdr, row1, col1, src.rows, src.cols);

	// down-right
	row=src.rows, col=src.cols;
	xcv_point_rotate (center.y, center.x, row, col, &row1, &col1, angle, is_clockwise);
	pt_src[2] = Point2f (col, row);
	pt_dst[2] = Point2f (col1, row1);
	border_check (&bdr, row1, col1, src.rows, src.cols);

	// down-left
	row=src.rows, col=0;
	xcv_point_rotate (center.y, center.x, row, col, &row1, &col1, angle, is_clockwise);
	pt_src[3] = Point2f (col, row);
	pt_dst[3] = Point2f (col1, row1);
	border_check (&bdr, row1, col1, src.rows, src.cols);

	// determine border color
	determine_border_color (src, &b, &g, &r);
	Scalar bc = Scalar (b, g, r);
	//copyMakeBorder (src, ext_src, bdr.u, bdr.d, bdr.l, bdr.r, BORDER_CONSTANT, bc);
	copyMakeBorder (src, ext_src, bdr.u, bdr.d, bdr.l, bdr.r, BORDER_CONSTANT, Scalar(255,255,255));

	// fix pt_src and pt_dst to new extended picture
	for (i=0; i<4; ++i) {
		pt_src[i].x += (float) bdr.l; // column
		pt_dst[i].x += (float) bdr.l; // column
		pt_src[i].y += (float) bdr.u; // row
		pt_dst[i].y += (float) bdr.u; // row
	}
	center.x += (float) bdr.l; // column
	center.y += (float) bdr.u; // row

	// rotate
	Mat tm = getPerspectiveTransform (pt_src, pt_dst);
	warpPerspective (ext_src, dst, tm, Size(ext_src.cols,ext_src.rows), INTER_LINEAR, BORDER_CONSTANT, Scalar(255,255,255));

	return 0;
}

void
xcv_imread (cv::Mat & img, const char * pic)
{
  img = imread (pic);
  if (img.empty())
    err_mesg ("'%s' is empty!", pic);
}

void
xcv_imread_gray (cv::Mat & img, const char * pic)
{
  Mat img0;

  img0 = imread (pic);
  if (img0.empty())
    err_mesg ("'%s' is empty!", pic);
  cvtColor (img0, img, COLOR_BGR2GRAY);
}

void
xcv_crop_3b (cv::Mat & img_orig, cv::Mat & img_new,
    int32_t col_beg, int32_t row_beg, int32_t col_ext, int32_t row_ext)
{
  unsigned char * dst;
  unsigned char * src;
  int i, j;
  int type;

  type = img_orig.type ();
  assert (type==CV_8UC3 || type==CV_8SC3);

  if (col_beg+col_ext > img_orig.cols)
    col_ext = img_orig.cols - col_beg;
  if (row_beg+row_ext > img_orig.rows)
    row_ext = img_orig.rows - row_beg;

  img_new = Mat::zeros (Size(col_ext,row_ext), type);
  for (i=0; i<row_ext; ++i) {
    for (j=0; j<col_ext; ++j) {
      img_new.at<Vec3b>(i,j) = img_orig.at<Vec3b>(row_beg+i,col_beg+j);
    }
  }
}

void
xcv_mean_ub (cv::Mat & img, unsigned char * b, unsigned char * g, unsigned char * r,
    int32_t col_beg, int32_t row_beg, int32_t col_ext, int32_t row_ext)
{
  int cn, type, depth;
  int32_t i, j, col_end, row_end;

  type = img.type ();
  depth = type & CV_MAT_DEPTH_MASK;
  assert (depth == CV_8U);
  cn = (type>>CV_CN_SHIFT) + 1;
}

void
xcv_fill_ub (cv::Mat & img, unsigned char b, unsigned char g, unsigned char r,
    int32_t col_beg, int32_t row_beg, int32_t col_ext, int32_t row_ext)
{

}

void
xcv_bgr2rgb_3b (cv::Mat & img)
{
  int i, j;
  int tmp;

  for (i=0; i<img.rows; ++i) {
    for (j=0; j<img.cols; ++j) {
      tmp = img.at<Vec3b>(i,j)[0];
      img.at<Vec3b>(i,j)[0] = img.at<Vec3b>(i,j)[2];
      img.at<Vec3b>(i,j)[2] = tmp;
    }
  }
}

typedef struct {
  Mat * img;
  Point * pt_start;
  vector<Point> * contour;
  vector<vector<Point>> * contours;
} mouse_data_t;

static int MAX_DIM = 1000;

static void
on_mouse (int event, int x, int y, int flags, void * data)
{
  mouse_data_t * dat = (mouse_data_t *) data;
  Mat & img = *(dat->img);
  vector<Point> & contour = *(dat->contour);
  vector<vector<Point>> & contours = *(dat->contours);
  Point cur_pt;

  if (event == CV_EVENT_LBUTTONDOWN) {
    printf ("x: %d\ty: %d\n", x, y);
    cur_pt = Point (x, y);
    dat->pt_start = &cur_pt;
    contour.clear ();
    contour.push_back (cur_pt);
    circle (img, cur_pt, 1, Scalar(0,0,255), -1, 8);
    imshow ("bin1.gray", img);
  } else if ((event==CV_EVENT_MOUSEMOVE) && (flags&CV_EVENT_FLAG_LBUTTON)) {
    printf ("x: %d\ty: %d\n", x, y);
    cur_pt = Point (x, y);
    line (img, contour.back(), cur_pt, Scalar(0,255,0), 1, 8, 0);
    circle (img, cur_pt, 1, Scalar(0,0,255), -1, 8);
    imshow ("bin1.gray", img);
    contour.push_back (cur_pt);
  } else if (event == CV_EVENT_LBUTTONUP) {
    printf ("x: %d\ty: %d\n", x, y);
    cur_pt = Point (x, y);
    line (img, *(dat->pt_start), cur_pt, Scalar(0,255,0), 1, 8, 0);
    circle (img, cur_pt, 1, Scalar(0,0,255), -1, 8);
    imshow ("bin1.gray", img);
    contour.push_back (cur_pt);
    contours.push_back (contour);
  }
}

void
xcv_manual_roi (const char * img_file, vector<vector<Point>> & contours)
{
  int n_cols;
  int n_rows;
  Mat img;
  mouse_data_t dat;
  vector<Point> contour;

  xcv_imread (img, img_file);
  dat.img = &img;
  dat.contour = &contour;
  dat.contours = &contours;
  namedWindow ("Manual ROI");

  if (img.cols > img.rows) {
    if (img.cols > MAX_DIM) {
      n_cols = MAX_DIM;
      n_rows = img.rows * img.cols / MAX_DIM;
    } else {
      n_cols = img.cols;
      n_rows = img.rows;
    }
  } else {
    if (img.rows > MAX_DIM) {
      n_rows = MAX_DIM;
      n_cols = img.cols * img.rows / MAX_DIM;
    } else {
      n_cols = img.cols;
      n_rows = img.rows;
    }
  }
  resizeWindow ("Manual ROI", n_cols, n_rows);

  setMouseCallback ("Manual ROI", on_mouse, &dat);
  imshow ("Manual ROI", img);
  waitKey (0);
}

void
xcv_legend (cv::Mat & img, int32_t row_beg, int32_t row_ext,
			             int32_t col_beg, int32_t col_ext,
									 std::vector<cv::Vec3b> & origin_colors, double max_val,
									 const char * tag)
{
	int32_t i, j, k;
	int32_t n_sep;
	int32_t step;
	int32_t beg, end;
	Vec3b color;
	vector<Vec3b> colors;

	assert (row_beg+row_ext < img.rows);
	assert (col_beg+col_ext < img.cols);

	n_sep = origin_colors.size() - 1;
	for (i=n_sep; i>=0; --i)
		colors.push_back (origin_colors[i]);

	row_ext = row_ext / n_sep * n_sep;
	step = row_ext / n_sep;

	end = 0;
	for (i=0; i<n_sep; ++i) {
		beg = end;
		end = beg + step;
		//printf ("%d-%d-%d\n", colors[i][0], colors[i][1], colors[i][2]);
		for (j=beg; j<end; ++j) {
			color[0] = colors[i][0] + (colors[i+1][0]-colors[i][0]) * (j-beg) / step;
			color[1] = colors[i][1] + (colors[i+1][1]-colors[i][1]) * (j-beg) / step;
			color[2] = colors[i][2] + (colors[i+1][2]-colors[i][2]) * (j-beg) / step;
			for (k=0; k<col_ext/2; ++k)
				img.at<Vec3b>(row_beg+j,col_beg+k) = color;
		}
	}

	int n_digit = (int) log10 (max_val);
	double base = pow (10, n_digit);
	double upper, lower;
	int n_scale = (int) (max_val / base);
	printf ("max_val: %.2lf\tbase: %.2lf\tscale: %d\n", max_val, base, n_scale);
	assert (n_scale >= 0 && n_scale < 10);
	if (n_scale > 1) {
		n_scale = n_scale / 2 * 2;
		upper = base * n_scale;
		lower = upper / 2;
	} else if (n_scale == 1) {
		upper = base;
		lower = base/2;
		if (lower < 1)
			lower = -1;
	} else {
		upper = max_val;
		lower = -1;
	}

	char buf[4096];
	int baseline;
	double scale;

	//printf ("%.2lf\t%.2lf\t%.2lf\n", lower, upper, max_val);

	int upper_row_idx;
	int lower_row_idx;
	int zero_row_idx;
	Size upper_text_blk;
	Size lower_text_blk;
	Size zero_text_blk;

	// upper
	upper_row_idx = row_beg + row_ext - (int)(upper/max_val*row_ext);
	if (NULL != tag) {
		if (upper > 1)
			sprintf (buf, "%.0lf%s", upper, tag);
		else
			sprintf (buf, "%.1lf%s", upper, tag);
	} else {
		if (upper > 1)
			sprintf (buf, "%.0lf", upper);
		else
			sprintf (buf, "%.1lf", upper);
	}
	string ulg (buf);
	upper_text_blk = getTextSize (ulg, FONT_HERSHEY_SIMPLEX, 1, 2, &baseline);
	scale = (double)(step) / (double)upper_text_blk.height;
	int col_left = img.cols - (col_beg+col_ext/2+10) - 10;
	double scale_width = col_left / (double)upper_text_blk.width;
	//printf ("col_left: %d\tscale: %.2lf\twidth_scale: %.2lf\n", col_left, scale, scale_width);
	if (scale_width < scale)
		scale = scale_width;

	// lower
	string llg;
	if (lower > 0) {
		lower_row_idx = row_beg + row_ext - (int)(lower/max_val*row_ext);
		if (NULL != tag)
			sprintf (buf, "%.0lf%s", lower, tag);
		else
			sprintf (buf, "%.0lf", lower);
		llg = string (buf);
		lower_text_blk = getTextSize (llg, FONT_HERSHEY_SIMPLEX, 1, 2, &baseline);
	}

	// zero
	zero_row_idx = row_beg + row_ext;
	sprintf (buf, "0");
	string zlg (buf);
	zero_text_blk = getTextSize (zlg, FONT_HERSHEY_SIMPLEX, 1, 2, &baseline);

	scale *= lg_text_scale;
	putText (img, ulg, Point(col_beg+col_ext/2+10,upper_row_idx), FONT_HERSHEY_SCRIPT_SIMPLEX, scale, Scalar(255,255,255), 10);
	if (lower > 0)
		putText (img, llg, Point(col_beg+col_ext/2+10,lower_row_idx), FONT_HERSHEY_SCRIPT_SIMPLEX, scale, Scalar(255,255,255), 10);
	putText (img, zlg, Point(col_beg+col_ext/2+10,zero_row_idx),  FONT_HERSHEY_SCRIPT_SIMPLEX, scale, Scalar(255,255,255), 10);
}

void
xcv_title (cv::Mat & img, int32_t title_row_beg, int32_t title_height, const char * title_content)
{
	int col_beg;
	int row_beg;
	int baseline;
	double scale;
	double width_scale;
	string title (title_content);
	Size blk_size = getTextSize (title, FONT_HERSHEY_SIMPLEX, 1, 2, &baseline);

	scale = (double)title_height / blk_size.height;
	width_scale = (double)(img.cols*8/10) / blk_size.width;
	if (width_scale < scale)
		scale = width_scale;

	col_beg = img.cols/2 - (int)(scale*blk_size.width)/2;
	row_beg = title_row_beg + (int)(scale*blk_size.height);
	putText (img, title, Point(col_beg,row_beg), FONT_HERSHEY_SIMPLEX, scale, Scalar(255,255,255), 3);
}

void
xcv_test_palette (const char * out_pic, std::vector<cv::Vec3b> & colors)
{
	int nrow;
	int ncol;
	int idx;
	int i, j;
	Mat img;
	vector<Point> pts;

	nrow = (colors.size()+1) / 10 + 1;
	ncol = 10;

	img = Mat::zeros (Size(ncol*100,nrow*100), CV_8UC3);

	for (i=0,idx=0; i<nrow; ++i) {
		for (j=0; j<ncol; ++j) {
			pts.clear ();
			pts.push_back (Point(j*100,i*100));
			pts.push_back (Point(j*100,i*100+100));
			pts.push_back (Point(j*100+100,i*100+100));
			pts.push_back (Point(j*100+100,i*100));
			fillConvexPoly (img, pts, colors[idx]);
			++idx;
		}
	}
	imwrite (out_pic, img);
}

void
load_colors (const char * palette_file, std::vector<cv::Vec3b> & colors)
{
	char line[4096];
	unsigned int num;
	FILE * in = ckopen (palette_file, "r");

	while (fgets(line,4096,in)) {
		assert (*line == '#');
		sscanf (line+1, "%x", &num);

		int b = num & 0xff;
		int g = (num>>8) & 0xff;
		int r = (num>>16) & 0xff;
		colors.push_back (Vec3b(b, g, r));
	}
}

void
load_palette_bgr (std::vector<cv::Vec3b> & colors)
{
	int i;
	unsigned int num;

	for (i=0; i<CV_PALETTE_NCOLORS; ++i) {
		num = xcv_palette_rgb[i];
		int r = (num>>16) & 0xff;
		int g = (num>>8) & 0xff;
		int b = num & 0xff;
		colors.push_back (Vec3b(b, g, r));
	}
}

void
load_palette_file (std::vector<cv::Vec3b> & colors, const char * file)
{
	char line[4096];
	char * ch;
	unsigned int num;
	FILE * in = ckopen (file, "r");

	while (fgets(line,4096,in)) {
		assert ((ch = strrchr(line,'#')) != NULL);
		sscanf (ch+1, "%x", &num);
		int r = (num>>16) & 0xff;
		int g = (num>>8) & 0xff;
		int b = num & 0xff;
		colors.push_back (Vec3b(b, g, r));
	}

	fclose (in);
}

void
xcv_scale_bar (cv::Mat & img, int32_t row, int32_t col, int32_t len, const char * tag)
{
	//int wid = 3;
	//int bl;
	//int th;
	//double scale;
	//Size tblk;

	//string lg (tag);
	//tblk = getTextSize (lg, FONT_HERSHEY_SIMPLEX, 1, 2, &bl);
	//scale = (double) len / tblk.width;
	//putText (img, lg, Point(col,row), FONT_HERSHEY_SIMPLEX, scale, Scalar(0,0,0), wid);

	//th = (int) tblk.height * scale;
	//line (img, Point(col,row+th+th/6), Point(col+len,row+th+th/6), Scalar(0,0,0), 3);
	//line (img, Point(col,row+th), Point(col,row+th+th/6), Scalar(0,0,0), 3);
	//line (img, Point(col+len,row+th), Point(col+len,row+th+th/6), Scalar(0,0,0), 3);
	line (img, Point(col,row), Point(col+len,row), Scalar(0,0,0), 30);
}

xcv_pal_t *
xcv_pal_init (void)
{
	xcv_pal_t * pal;
 
	pal = (xcv_pal_t *) ckmalloc (sizeof(xcv_pal_t));
	pal->n = 0;
	pal->m = 16;
	pal->r = (int *) ckalloc (pal->m, sizeof(int));
	pal->g = (int *) ckalloc (pal->m, sizeof(int));
	pal->b = (int *) ckalloc (pal->m, sizeof(int));

	return pal;
}

void
xcv_pal_clear (xcv_pal_t * pal)
{
	pal->n = 0;
}

void
xcv_pal_free (xcv_pal_t * pal)
{
	free (pal->r);
	free (pal->g);
	free (pal->b);
	free (pal);
}

void
xcv_pal_resize (xcv_pal_t * pal, int size)
{
	if (size <= pal->m)
		return;

	pal->m = size;
	pal->r = (int *) ckrealloc (pal->r, pal->m*sizeof(int));
	pal->g = (int *) ckrealloc (pal->r, pal->m*sizeof(int));
	pal->b = (int *) ckrealloc (pal->r, pal->m*sizeof(int));
}

void
xcv_pal_add (xcv_pal_t * pal, int r, int g, int b)
{
	if (pal->n+1 > pal->m)
		xcv_pal_resize (pal, 2*pal->m);

	pal->r[pal->n] = r;
	pal->g[pal->n] = g;
	pal->b[pal->n] = b;
	++pal->n;
}

void
xcv_create_pal (xcv_pal_t * src_pal, xcv_pal_t * dst_pal, int n_dst_colors)
{
	xcv_pal_resize (dst_pal, n_dst_colors);
}
