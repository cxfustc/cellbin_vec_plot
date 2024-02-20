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
		fprintf (stderr, "         -n   INT   Minimum cell counts of annotated region (default: >=10)\n");
		fprintf (stderr, "         -m   INT   Minimum area of cells (default: >=50)\n");
		fprintf (stderr, "         -M   INT   Maximum area of cells (default: <=10000)\n");
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
	fprintf (fout, "#%d\tMinimum cell counts in annotated regions\n", min_num_cells);
	fprintf (fout, "#%d\tMinimum area of each cell\n", min_cell_area);
	fprintf (fout, "#%d\tMaximum area of each cell\n", max_cell_area);
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

typedef struct {
	int gid;
	int x, y;
	int cnt;
} spot_t;
MP_DEF (spot, spot_t);

static int
extr_main (int argc, char * argv[])
{
	if (argc < 3) {
		fprintf (stderr, "\n");
		fprintf (stderr, "Usage:   cellbin_vec_plot pre_ann extr [options] <gem> <mask.tif>\n");
		fprintf (stderr, "Contact: chenxi1@genomics.cn\n");
		fprintf (stderr, "Options: -o   STR   Output directory (default: ./)\n");
		fprintf (stderr, "         -s   STR   Output sample name (default: sample)\n");
		fprintf (stderr, "         -f         Fix min X/Y to 0 (default: not fixed)\n");
		fprintf (stderr, "\n");
		return 1;
	}

	char buf[4096];
	char gene[4096];
	int copt;
	int i, j;
	int label;
	int fix_xy;
	int xmax;
	int ymax;
	int cell_type;
	int32_t n_cls;
	int32_t n_genes;
	int32_t max_cells;
	int32_t r, g, b;
	int32_t n_labels;
	int32_t num_cells;
	int32_t pre_num_cells;
	uint32_t num;
	int * expr;
	int * expr_ptr;
	int * x;
	int * y;
	FILE * in;
	FILE * out;
	gzFile gin;
	gzFile gout;
	Mat mask;
	Mat labels;
	Mat stats;
	Mat centroids;
	Mat ann_mask;
	str_t s;
	str_t * str_ptr;
	str_hash_t * gene_hash;
	str_set_t * gene_names;
	xh_item_t * xh_ptr;
	spot_t * spot;
	mp_t(spot) * spots;

	fix_xy = 0;
	while ((copt = getopt(argc,argv,"o:s:f")) != -1) {
		if (copt == 'o')
			strcpy (out_dir, optarg);
		else if (copt == 's')
			strcpy (spl_name, optarg);
		else if (copt == 'f')
			fix_xy = 1;
	}

	if (optind >= argc)
		err_mesg ("fail to get input gem file!");
	else
		strcpy (gem_file, argv[optind++]);

	if (optind >= argc)
		err_mesg ("fail to get mask image!");
	else
		strcpy (mask_img, argv[optind++]);

	gin = ckgzopen (gem_file, "r");
	gene_hash = str_hash_init ();
	spots = mp_init (spot, NULL);
	while (gzgets(gin,buf,4096))
		if (*buf != '#')
			break;
	while (gzgets(gin,buf,4096)) {
		spot = mp_alloc (spot, spots);
		sscanf (buf, "%s %d %d %d", gene, &spot->x, &spot->y, &spot->cnt);
		s.s = gene;
		s.l = strlen (gene);
		xh_ptr = str_hash_add3 (gene_hash, &s);
		spot->gid = xh_ptr->id;
	}
	gzclose (gin);

	gene_names = str_set_init ();
	xh_set_key_iter_init (xstr, gene_hash);
	while ((str_ptr = xh_set_key_iter_next(xstr,gene_hash)) != NULL)
		str_set_add2 (gene_names, str_ptr->s, str_ptr->l);

	n_genes = gene_names->n;
	max_cells = 10000;
	expr = (int *)ckmalloc (n_genes * max_cells * sizeof(int));
	x = (int *) ckmalloc (max_cells * sizeof(int));
	y = (int *) ckmalloc (max_cells * sizeof(int));

	sprintf (buf, "%s/%s.prep.txt", out_dir, spl_name);
	in = ckopen (buf, "r");

	sprintf (buf, "%s/%s.cluster_info.txt", out_dir, spl_name);
	out = ckopen (buf, "w");
	fprintf (out, "Cluster\tColor\tCell_cnts\n");

	sprintf (buf, "%s/%s.pre_anno.gem.gz", out_dir, spl_name);
	gout = ckgzopen (buf, "w");
	gzprintf (gout, "#FileFormat=GEMv0.1\n");
	gzprintf (gout, "#OffsetX=0\n#OffsetY=0\n");
	gzprintf (gout, "geneID\tx\ty\tMIDCounts\tlabel\tcellType\n");

	xcv_imread (mask, mask_img);

	if (fix_xy) {
		int xmin = INT_MAX;
		int ymin = INT_MAX;
		for (i=0; i<spots->n; ++i) {
			spot = mp_at (spot, spots, i);
			if (spot->x < xmin) xmin = spot->x;
			if (spot->y < ymin) ymin = spot->y;
		}
		for (i=0; i<spots->n; ++i) {
			spot = mp_at (spot, spots, i);
			spot->x -= xmin;
			spot->y -= ymin;
		}
	}

	xmax = -1;
	ymax = -1;
	for (i=0; i<spots->n; ++i) {
		spot = mp_at (spot, spots, i);
		if (spot->x > xmax) xmax = spot->x;
		if (spot->y > ymax) ymax = spot->y;
	}
	if (ymax >= mask.rows)
		err_mesg ("Coordinates in gem file exceeds mask image: GEM/y %d - image rows %d", ymax, mask.rows);
	if (xmax >= mask.cols)
		err_mesg ("Coordinates in gem file exceeds mask image: GEM/x %d - image columns %d", xmax, mask.cols);

	cell_type = 0;
	num_cells = 0;
	pre_num_cells = 0;
	while (fgets(buf,4096,in))
		if (*buf != '#')
			break;
	while (fgets(buf,4096,in)) {
		sscanf (buf, "%*s %x", &num);
		r = (num >> 16) & 0xff;
		g = (num >> 8 ) & 0xff;
		b =  num        & 0xff;

		Mat ann_mask = Mat::zeros (mask.size(), CV_8UC1);
		for (i=0; i<mask.rows; ++i)
			for (j=0; j<mask.cols; ++j) {
				Vec3b v = mask.at<Vec3b>(i,j);
				if (v[2]!=r || v[1]!=g || v[0]!=b)
					continue;
				ann_mask.at<uchar>(i,j) = 255;
			}

		n_labels = connectedComponentsWithStats (ann_mask, labels, stats, centroids, 4, CV_32S);

		if (n_labels >= max_cells) {
			max_cells = ((n_labels>>8)+1) << 8;
			free (expr); free (x); free (y);
			expr = (int *)ckmalloc (n_genes * max_cells * sizeof(int));
			x = (int *) ckmalloc (max_cells * sizeof(int));
			y = (int *) ckmalloc (max_cells * sizeof(int));
		}
		memset (expr, 0, n_genes*max_cells);
		for (i=0; i<max_cells; ++i)
			x[i] = -1;

		for (i=0; i<spots->n; ++i) {
			spot = mp_at (spot, spots, i);
			label = labels.at<int>(spot->y,spot->x);
			if (label <= 0)
				continue;

			assert (label < max_cells);

			if (x[label] < 0) {
				x[label] = spot->x;
				y[label] = spot->y;
			}
			expr[label*n_genes + spot->gid] += spot->cnt;
		}

		for (i=1; i<n_labels; ++i) {
			if (x[i] < 0)
				continue;
			expr_ptr = expr + i * n_genes;
			for (j=0; j<n_genes; ++j) {
				if (expr_ptr[j] <= 0)
					continue;

				gzprintf (gout, "%s\t%d\t%d\t%d\t%d\t%d\n",
						gene_names->pool[j].s,
						x[i], y[i],
						expr_ptr[j],
						num_cells+1,
						cell_type+1);
			}
			++num_cells;
		}
		fprintf (out, "%d\t%06X\t%d\n", cell_type+1, num, num_cells-pre_num_cells);
		printf ("Cluster %d, Valid cell counts: %d\n", cell_type+1, num_cells-pre_num_cells);
		++cell_type;
		pre_num_cells = num_cells;
	}
	printf ("Total cell counts: %d\n", num_cells);

	fclose (in);
	fclose (out);
	gzclose (gout);

	return 0;
}

int
pre_anno_gem_main (int argc, char * argv[])
{
	if (argc < 2) {
		fprintf (stderr, "\n");
		fprintf (stderr, "Usage:   cellbin_vec_plot pre_ann <sub_cmd>\n");
		fprintf (stderr, "Contact: chenxi1@genomics.cn\n");
		fprintf (stderr, "Sub_Cmd: prep   Prepare for annotated cellbin gem extraction\n");
		fprintf (stderr, "         extr   Extract annotated cellbin gem\n");
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
	max_cell_area = 10000;
	min_num_cells = 10;

	strcpy (out_dir, ".");
	strcpy (spl_name, "sample");

	if (strcmp(argv[1],"prep") == 0)
		ret = prep_main (argc-1, argv+1);
	else if (strcmp(argv[1],"extr") == 0)
		ret = extr_main (argc-1, argv+1);
	else
		err_mesg ("Invalid sub-command: %s!", argv[1]);

	return ret;
}
