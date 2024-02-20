#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "mp.h"
#include "hash.h"
#include "str.h"
#include "str_hash.h"
#include "utils.h"
#include "image.h"
#include "array.h"

using namespace cv;
using namespace std;

static char * gem_file;
static char * mask_img;
static char * spl_name;
static char * out_dir;
static uint32_t min_area;
static int32_t min_cell_area;
static int32_t max_cell_area;
static int32_t min_num_cells;

static int
plot_main (int argc, char * argv[])
{
	if (argc < 3) {
		fprintf (stderr, "Usage: cellbin_vec_plot lasso plot <cell.txt> <out.tif>\n");
		return 1;
	}

	char buf[4096];
	char color[4096];
	char * ch;
	int xmax, ymax;
	int x, y;
	int r, g, b;
	int rad = 10;
	uint32_t num;
	float xv, yv;
	FILE * in = ckopen (argv[1], "r");
	Mat img;

	xmax = ymax = INT_MIN;
	fgets (buf, 4096, in);
	while (fgets(buf, 4096, in)) {
		sscanf (buf, "%*s %f %f", &xv, &yv);
		x = (int) xv;
		y = (int) yv;
		if (x > xmax) xmax = x;
		if (y > ymax) ymax = y;
	}
	img = Mat::zeros (Size(xmax+1,ymax+1), CV_8UC3);

	rewind (in);
	fgets (buf, 4096, in);
	while (fgets(buf, 4096, in)) {
		sscanf (buf, "%*s %f %f %*s %s", &xv, &yv, color);
		x = (int) xv;
		y = (int) yv;
		y = ymax - y;
		ch = color;
		if (*ch == '#')
			ch += 1;
		sscanf (ch, "%x", &num);

		r = (num >> 16) & 0xff;
		g = (num >> 8 ) & 0xff;
		b =  num        & 0xff;

		circle (img, Point(x,y), rad, Scalar(b,g,r), -1, 8);
	}
	fclose (in);

	if (str_cmp_tail(argv[2],".tif",4) != 0)
		sprintf (buf, "%s.tif", argv[2]);
	else
		strcpy (buf, argv[2]);
	imwrite (buf, img);

	return 0;
}

static int
prep_main (int argc, char * argv[])
{
	if (argc < 2) {
		fprintf (stderr, "\n");
		fprintf (stderr, "Usage:   cellbin_vec_plot pre_ann prep [options] <mask.tif>\n");
		fprintf (stderr, "Contact: chenxi1@genomics.cn\n");
		fprintf (stderr, "Options: -b   STR   RGB value(s) of the backgroud of image, separated by comma (,) (defult: null)\n");
		fprintf (stderr, "         -o   STR   Output directory (default: ./)\n");
		fprintf (stderr, "         -s   STR   Output sample name (default: sample)\n");
		fprintf (stderr, "         -r   INT   Minimum area (number of DNBs) of annotated region (default: >=1000)\n");
		fprintf (stderr, "         -n   INT   Minimum mumber of parts of annotated region (default: >=1)\n");
		fprintf (stderr, "         -m   INT   Minimum area of parts (default: >=50)\n");
		fprintf (stderr, "         -M   INT   Maximum area of parts (default: <=1000000)\n");
		fprintf (stderr, "\n");
		return 1;
	}

	char buf[4096];
	char * ch;
	int copt;
	int i, j;
	int z;
	int area;
	int label;
	int idx;
	int n_labels;
	int n_valid_cells;
	int32_t color;
	int32_t r, g, b;
	int32_t * i32_ptr;
	uint32_t num;
	FILE * fout;
	Mat mask;
	Mat labels;
	Mat stats;
	Mat centroids;
	arr_t(i32) * colors;
	arr_t(i32) * cell_area;
	arr_t(u32) * color_area;
	xh_set_t(i32) * color_stat;
	xh_set_t(i32) * color_hash;
	xh_set_t(i32) * bg_color_hash;

	bg_color_hash = xh_i32_set_init ();
	while ((copt = getopt(argc,argv,"b:o:s:r:n:m:M:")) != -1) {
		if (copt == 'b') {
			strcpy (buf, optarg);
			assert ((ch = strtok(buf,",")) != NULL);
			if (*ch == '#')
				sscanf (ch+1, "%x", &num);
			else
				sscanf (ch, "%x", &num);
			color = num;
			xh_set_add (i32, bg_color_hash, &color);

			while ((ch = strtok(NULL,",")) != NULL) {
				if (*ch == '#')
					sscanf (ch+1, "%x", &num);
				else
					sscanf (ch, "%x", &num);
				color = num;
				xh_set_add (i32, bg_color_hash, &color);
			}
		} else if (copt == 'o')
			strcpy (out_dir, optarg);
		else if (copt == 's')
			strcpy (spl_name, optarg);
		else if (copt == 'r')
			min_area = atoi (optarg);
		else if (copt == 'n')
			min_num_cells = atoi (optarg);
		else if (copt == 'm')
			min_cell_area = atoi (optarg);
		else if (copt == 'M')
			max_cell_area = atoi (optarg);
	}

	if (optind >= argc)
		err_mesg ("fail to get mask image!");
	else
		strcpy (mask_img, argv[optind++]);

	sprintf (buf, "mkdir -p %s/prep_images", out_dir);
	system (buf);

	color_stat = xh_i32_set_init ();
	xcv_imread (mask, mask_img);
	for (i=0; i<mask.rows; ++i)
		for (j=0; j<mask.cols; ++j) {
			Vec3b v = mask.at<Vec3b>(i,j);
			color = (v[2]<<16) | (v[1]<<8) | v[0];

			// skip backgroud colors
			if (xh_set_search(i32,bg_color_hash,&color) == XH_EXIST)
				continue;

			xh_set_add (i32, color_stat, &color);
		}

	colors = arr_init (i32);
	color_area = arr_init (u32);
	color_hash = xh_i32_set_init ();
	for (i=0; i<xh_set_cnt(color_stat); ++i) {
		i32_ptr = (int32_t *) color_stat->hash->pool[i].key;
		if (color_stat->hash->pool[i].multi < min_area)
			continue;

		xh_set_add (i32, color_hash, i32_ptr);
		arr_add (i32, colors, *i32_ptr);
		arr_add (u32, color_area, color_stat->hash->pool[i].multi);
	}

	sprintf (buf, "%s/%s.prep.txt", out_dir, spl_name);
	fout = ckopen (buf, "w");

	fprintf (fout, "#%d\tMinimum area of annotated regions\n", min_area);
	fprintf (fout, "#%d\tMinimum parts in annotated regions\n", min_num_cells);
	fprintf (fout, "#%d\tMinimum area of each part\n", min_cell_area);
	fprintf (fout, "#%d\tMaximum area of each part\n", max_cell_area);
	fprintf (fout, "Index\tColor\tTotal_Pixels\n");

	idx = 0;
	cell_area = arr_init (i32);
	for (z=0; z<colors->n; ++z) {
		Mat ann_mask = Mat::zeros (mask.size(), CV_8UC1);
		r = (colors->arr[z] >> 16) & 0xff;
		g = (colors->arr[z] >> 8 ) & 0xff;
		b = (colors->arr[z]      ) & 0xff;

		for (i=0; i<mask.rows; ++i)
			for (j=0; j<mask.cols; ++j) {
				Vec3b v = mask.at<Vec3b>(i,j);
				if (v[2]!=r || v[1]!=g || v[0]!=b)
					continue;
				ann_mask.at<uchar>(i,j) = 255;
			}
		n_labels = connectedComponentsWithStats (ann_mask, labels, stats, centroids, 4, CV_32S);
		if (n_labels < min_num_cells+1) // label0 for background
			continue;

		Mat out = Mat::zeros (mask.size(), CV_8UC3);

		if (r<64 && g<64 && b<64) {
			for (i=0; i<out.rows; ++i)
				for (j=0; j<out.cols; ++j)
					out.at<Vec3b>(i,j) = Vec3b (255,255,255);
		}

		n_valid_cells = 0;
		for (i=0; i<mask.rows; ++i)
			for (j=0; j<mask.cols; ++j) {
				label = labels.at<int>(i,j);
				if (label < 1)
					continue;
				area = stats.at<int>(label,4);
				if (area<min_cell_area || area>max_cell_area)
					continue;
				++n_valid_cells;
				out.at<Vec3b>(i,j) = mask.at<Vec3b>(i,j);
			}
		if (n_valid_cells < min_num_cells)
			continue;

		sprintf (buf, "%s/prep_images/%s.%d.mask.tif", out_dir, spl_name, idx+1);
		imwrite (buf, out);

		fprintf (fout, "%d\t%06X\t%d\n", idx+1, colors->arr[z], n_valid_cells);

		++idx;
	}
	fclose (fout);

	return 0;
}

static int
extr_main (int argc, char * argv[])
{
	if (argc < 4) {
		fprintf (stderr, "\n");
		fprintf (stderr, "Usage:   cellbin_vec_plot pre_ann lass [options] <cell.cord> <lasso.tif> <lasso.color>\n");
		fprintf (stderr, "Contact: chenxi1@genomics.cn\n");
		fprintf (stderr, "Options: -o   STR   Output directory (default: ./)\n");
		fprintf (stderr, "         -s   STR   Output sample name (default: sample)\n");
//		fprintf (stderr, "         -f         Fix min X/Y to 0 (default: not fixed)\n");
		fprintf (stderr, "\n");
		return 1;
	}

	char * ch;
	char sep[3];
	char buf[4096];
	int copt;
	int i, j;
	int r, g, b;
	int fix_xy;
	int sep_idx;
	uint32_t color;
	FILE * in;
	FILE * out;
	Mat mask;
	Mat lmask;

	while ((copt = getopt(argc,argv,"o:s:f")) != -1) {
		if (copt == 'o')
			strcpy (out_dir, optarg);
		else if (copt == 's')
			strcpy (spl_name, optarg);
		else if (copt == 'f')
			fix_xy = 1;
	}

	if (optind >= argc)
		err_mesg ("fail to open cell xy file!");
	else
		strcpy (gem_file, argv[optind++]);

	if (optind >= argc)
		err_mesg ("fail to get mask image!");
	else
		strcpy (mask_img, argv[optind++]);

	if (optind >= argc)
		err_mesg ("fail to get color of lasso!");
	else {
		ch = argv[optind++];
		if (*ch == '#')
			ch += 1;
		sscanf (ch, "%x", &color);

		r = (color >> 16) & 0xff;
		g = (color >> 8 ) & 0xff;
		b =  color        & 0xff;
	}

	xcv_imread (mask, mask_img);
	lmask = Mat::zeros (mask.size(), CV_8UC1);
	for (i=0; i<mask.rows; ++i)
		for (j=0; j<mask.cols; ++j) {
			Vec3b v = mask.at<Vec3b>(i,j);
			if (v[2]!=r || v[1]!=g || v[0]!=b)
				lmask.at<uchar>(i,j) = 255;
		}

	int n_labels;
	int area;
	int max_area;
	int bg_idx;
	int x, y;
	int xmax;
	int ymax;
	int label;
	int * lbl2idx;
	float vx, vy;
	Mat labels;
	Mat stats;
	Mat centroids;
	n_labels = connectedComponentsWithStats (lmask, labels, stats, centroids, 4, CV_32S);

	bg_idx = -1;
	max_area = 0;
	for (i=1; i<n_labels; ++i) {
		area = stats.at<int>(i,4);
		if (area > max_area) {
			max_area = area;
			bg_idx = i;
		}
	}

	Mat display = Mat::zeros (lmask.size(), CV_8UC3);
	for (i=0; i<lmask.rows; ++i)
		for (j=0; j<lmask.cols; ++j) {
			if (lmask.at<uchar>(i,j) == 255)
				display.at<Vec3b>(i,j) = Vec3b (255,255,255);
			else
				display.at<Vec3b>(i,j) = Vec3b (0,0,0);
		}
	int label_idx = 1;
	int baseline;
	lbl2idx = (int *) ckmalloc (n_labels*sizeof(int));
	for (i=0; i<n_labels; ++i)
		lbl2idx[i] = -1;
	for (i=1; i<n_labels; ++i) {
		if (i == bg_idx)
			continue;
		x = (int) centroids.at<double>(i,0);
		y = (int) centroids.at<double>(i,1);

		sprintf (buf, "%d", label_idx);
		string l (buf);
		Size blk = getTextSize (l, FONT_HERSHEY_SIMPLEX, 1, 2, &baseline);
		double scale = (double) lmask.rows/20 / blk.height;
		putText (display, l, Point(x,y), FONT_HERSHEY_SIMPLEX, scale, Scalar(0,0,255), 5);

		lbl2idx[i] = label_idx;
		label_idx += 1;
	}
	sprintf (buf, "%s/%s.region_label.tif", out_dir, spl_name);
	imwrite (buf, display);

	sep[0] = '\t';
	sep[1] = ' ';

	in = ckopen (gem_file, "r");
	sprintf (buf, "%s/%s.in_lasso.txt", out_dir, spl_name);
	out = ckopen (buf, "w");

	fgets (buf, 4096, in);
	chomp (buf);
	sep_idx = -1;
	for (ch=buf; *ch!='\0'; ++ch) {
		if (*ch == '\t') {
			sep_idx = 0;
			break;
		}
		if (*ch == ' ') {
			sep_idx = 1;
			break;
		}
	}
	if (sep_idx < 0)
		err_mesg ("file '%s' must be separated by '\\t' or ' '!", gem_file);
	fprintf (out, "%s%cpart\n", buf, sep[sep_idx]);

	xmax = ymax = INT_MIN;
	while (fgets(buf,4096,in)) {
		sscanf (buf, "%*s %f %f", &vx, &vy);
		x = (int) vx;
		y = (int) vy;
		if (x > xmax) xmax = x;
		if (y > ymax) ymax = y;
	}

	rewind (in);
	fgets (buf, 4096, in);
	while (fgets(buf,4096,in)) {
		chomp (buf);
		sscanf (buf, "%*s %f %f", &vx, &vy);
		x = (int) vx;
		y = (int) vy;
		y = ymax - y;
		label = labels.at<int>(y,x);
		if (label==0 || label==bg_idx)
			continue;
		fprintf (out, "%s%c%d\n", buf, sep[sep_idx], lbl2idx[label]);
	}

	fclose (in);
	fclose (out);

	char * params[3];
	int ret;
	params[0] = NULL;
	params[1] = ALLOC_LINE;
	params[2] = ALLOC_LINE;

	sprintf (params[1], "%s/%s.in_lasso.txt", out_dir, spl_name);
	sprintf (params[2], "%s/%s.in_lasso.tif", out_dir, spl_name);
	ret =  plot_main (3, params);

	return ret;
}

int
lasso_main (int argc, char * argv[])
{
	if (argc < 2) {
		fprintf (stderr, "\n");
		fprintf (stderr, "Usage:   cellbin_vec_plot lasso <sub_cmd>\n");
		fprintf (stderr, "Contact: chenxi1@genomics.cn\n");
		fprintf (stderr, "Sub_Cmd: plot   Plot image for lasso\n");
		fprintf (stderr, "         prep   Confrim color of lasso\n");
		fprintf (stderr, "         expr   Extract cells in lasso\n");
		fprintf (stderr, "\n");

		return 1;
	}

	int ret;

	gem_file = ALLOC_LINE;
	mask_img = ALLOC_LINE;
	spl_name = ALLOC_LINE;
	out_dir  = ALLOC_LINE;

	min_area = 1000;
	min_cell_area = 50;
	max_cell_area = 1000000;
	min_num_cells = 1;

	strcpy (out_dir, ".");
	strcpy (spl_name, "sample");

	if (strcmp(argv[1],"plot") == 0)
		ret = plot_main (argc-1, argv+1);
	else if (strcmp(argv[1],"prep") == 0)
		ret = prep_main (argc-1, argv+1);
	else if (strcmp(argv[1],"extr") == 0)
		ret = extr_main (argc-1, argv+1);
	else
		err_mesg ("Invalid sub-command: %s!", argv[1]);

	return ret;
}
