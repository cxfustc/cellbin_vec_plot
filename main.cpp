/***********************************************************
  *File Name: 
  *Description: 
  *Author: Chen Xi
  *Email: chenxi1@genomics.cn
  *Create Time: 2022-08-10 11:09:41
  *Edit History: 
***********************************************************/

#include <time.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "mp.h"
#include "str.h"
#include "hash.h"
#include "array.h"
#include "image.h"
#include "utils.h"
#include "str_hash.h"
#include "hash_func.h"

using namespace cv;
using namespace std;

extern int pre_anno_gem_main (int argc, char * argv[]);
extern int lasso_main (int argc, char * argv[]);

typedef struct {
	int r, g, b, blk;
} col_t;

typedef struct {
	int x, y;
	int tidx;
	int blk;
} bin_t;

MP_DEF (spot, bin_t);

static int idx2ros[4];
static int idx2cos[4];
static int next_start[4];

static char * cluster_file;
static char * color_file;
static char * expr_file;
static char * rds_file;
static char * gem_file;
static char * gene_list_file;
static char * out_dir;
static char * spl_name;
static char * mask_img;
static char * bg;
static char * text_color;
static char * all_cell_pfile;
static int l_bar;
static int data_type;
static int ex_xy;
static int run_norm;
static float max_expr;
static float hw_ratio;

static str_set_t * color_list;

/*
 *   0
 * 3 X 1
 *   2
 */

static int
cell_contour (Mat & img, Mat & drawed, int row, int col, arr_t(i32) * ctr, arr_t(i32) * ctc, FILE * out)
{
	uint8_t * ptr;
	int l, r, u, d;
	int i, j;
	int start;
	int pr, pc;
	int pr0, pc0;
	int pix[4];

	pr = row;
	pc = col;
	//assert (img.at<uchar>(pr,pc) == 255);
	if (img.at<uchar>(pr,pc) != 255)
		return -1;

	// find start point
	for ( ; pr>=0; --pr)
		if (img.at<uchar>(pr,pc) != 255)
			break;
	++pr;

	// skip cells that have been drawed
	if (drawed.at<uchar>(pr,pc) == 255)
		return 1;

	start = 0;
	pr0 = pr;
	pc0 = pc;

	arr_clear (i32, ctc);
	arr_clear (i32, ctr);

	for (;;) {
		// abnormal cell, too large
		if (ctr->n > 1000)
			return -1;

		l = (pc > 1);
		r = (pc < img.cols-1);
		u = (pr > 1);
		d = (pr < img.rows-1);

		memset (pix, 0, 4*sizeof(int));

		if (u) {
			ptr = img.data + (pr-1)*img.cols + pc;
			pix[0] = *ptr;
		}

		ptr = img.data + pr*img.cols + pc;
		if (l) { pix[3] = *(ptr-1); }
		if (r) { pix[1] = *(ptr+1); }

		if (d) {
			ptr = img.data + (pr+1)*img.cols + pc;
			pix[2] = *ptr;
		}

		for (i=0,j=start; i<4; ++i,++j) {
			j = j % 4;
			if (pix[j] == 255)
				break;
		}

		if (pix[j] != 255) {
			return -1;
//			assert (ctr->n == 0);
//			arr_add (i32, ctr, pr);
//			arr_add (i32, ctc, pc);
//			break;
		}

		arr_add (i32, ctr, pr);
		arr_add (i32, ctc, pc);

		pr = pr + idx2ros[j];
		pc = pc + idx2cos[j];
		start = next_start[j];

		if (pr==pr0 && pc==pc0) {
			arr_add (i32, ctr, pr);
			arr_add (i32, ctc, pc);
			break;
		}
	}

	for (i=0; i<ctr->n; ++i) {
		fprintf (out, "%d\t%d\n", ctr->arr[i], ctc->arr[i]);
		drawed.at<uchar>(ctr->arr[i],ctc->arr[i]) = 255;
	}

	return 0;
}

static void
set_text_color (void)
{
	char * ch;
	int r, g, b;
	int white_dist;
	int black_dist;
	uint32_t n;

	assert (*bg == '#');
	sscanf (bg+1, "%x", &n);

	r = (n>>16) & 0xff;
	g = (n>>8)  & 0xff;
	b =  n      & 0xff;

	white_dist = (255-r) + (255-g) + (255-b);
	black_dist = r + g + b;

	if (white_dist < black_dist)
		strcpy (text_color, "#000000");
	else
		strcpy (text_color, "#FFFFFF");
}

typedef struct {
	int row, col;
} pnt_t;
MP_DEF (pnt, pnt_t);

static int
cut_tissue_contour (const char * file)
{
	char line[4096];
	int i;
	int row_min;
	int col_min;
	FILE * in;
	FILE * out;
	pnt_t * p;
	mp_t(pnt) * pnts;

	in = ckopen (file, "r");
	pnts = mp_init (pnt, NULL);
	row_min = col_min = INT_MAX;
	while (fgets(line,4096,in)) {
		p = mp_alloc (pnt, pnts);
		sscanf (line, "%d %d", &p->row, &p->col);
		if (p->row < row_min) row_min = p->row;
		if (p->col < col_min) col_min = p->col;
	}
	fclose (in);

	out = ckopen (file, "w");
	for (i=0; i<pnts->n; ++i) {
		p = mp_at (pnt, pnts, i);
		fprintf (out, "%d\t%d\n",
				p->row-row_min, p->col-col_min);
	}
	fclose (out);

	return 0;
}

static int
cellbin_cluster_plot (const char * prefix)
{
	char line[4096];
	char file[4096];
	char buff[4096];
	char * ch;
	int val;
	int i, j;
	int x, y;
	int cls;
	int l;
	int n_plot;
	uint32_t n;
	int64_t cnt;
	FILE * in;
	FILE * out;
	FILE * idx_out;
	xh_item_t * xh_ptr;
	Mat img;
	Mat drawed;
	str_t s;
	str_t * sp;
	str_t * col_str;
	mp_t(spot) * bins;
	bin_t * bin;
	str_set_t * tnames;
	str_hash_t * types;
	col_t * colors;
	arr_t(i32) * ctr;
	arr_t(i32) * ctc;

	in = ckopen (cluster_file, "r");
	bins = mp_init (spot, NULL);
	types = str_hash_init ();
	while (fgets(line, 4096, in))
		if (*line != '#')
			break;
	while (fgets(line, 4096, in)) {
		chomp (line);

		assert ((ch = strtok(line, "\t"))!=NULL);
		x = atoi (ch);
		assert ((ch = strtok(NULL, "\t"))!=NULL);
		y = atoi (ch);

		assert ((ch = strtok(NULL, "\t"))!=NULL);
		for ( ; *ch!='\0'; ++ch)
			if (!isspace(*ch))
				break;
		l = strlen (ch);
		for (i=l-1; i>=0; --i)
			if (!isspace(ch[i]))
				break;
		ch[i+1] = '\0';

		s.s = ch;
		s.l = strlen (ch);
		xh_ptr = str_hash_add3 (types, &s);

		bin = mp_alloc (spot, bins);
		bin->x = x;
		bin->y = y;
		bin->tidx = xh_ptr->id;
	}
	fclose (in);

	tnames = str_set_init ();
	xh_set_key_iter_init (xstr, types);
	while ((sp = xh_set_key_iter_next(xstr,types)) != NULL)
		str_set_add2 (tnames, sp->s, sp->l);

	colors = (col_t *) ckalloc (tnames->n, sizeof(col_t));
	col_str = (str_t *) ckalloc (tnames->n, sizeof(str_t));
	in = ckopen (color_file, "r");
	while (fgets(line,4096,in)) {
		assert ((ch = strrchr(line,'#')) != NULL);
		sscanf (ch+1, "%x", &n);
		sscanf (ch, "%s", buff);

		for (--ch; ch>=line; --ch)
			if (!isspace(*ch))
				break;
		*(ch+1) = '\0';
		for (ch=line; *ch!='\0'; ++ch)
			if (!isspace(*ch))
				break;
		s.s = ch;
		s.l = strlen (ch);
		xh_ptr = str_hash_search3 (types, &s);

		if (xh_ptr == NULL)
			continue;

		colors[xh_ptr->id].b = n & 0xff;
		colors[xh_ptr->id].g = (n>>8) & 0xff;
		colors[xh_ptr->id].r = (n>>16) & 0xff;

		str_assign (col_str+xh_ptr->id, buff);
	}
	fclose (in);

	for (i=0; i<tnames->n; ++i)
		if (colors[i].b==0 && colors[i].g==0 && colors[i].r==0)
			err_mesg ("Color of luster '%s' has not set!", tnames->pool[i].s);

	xcv_imread_gray (img, mask_img);
	for (i=0; i<img.rows; ++i)
		for (j=0; j<img.cols; ++j)
			if (img.at<uchar>(i,j) > 0)
				img.at<uchar>(i,j) = 255;

	drawed = Mat::zeros (img.size(), CV_8UC1);

	cnt = 0;
	n_plot = 0;
	ctr = arr_init (i32);
	ctc = arr_init (i32);
	sprintf (file, "%s.cell_contour.txt", prefix);
	out = ckopen (file, "w");
	sprintf (file, "%s.cell_contour.idx.txt", prefix);
	idx_out = ckopen (file, "w");
	for (i=0; i<bins->n; ++i) {
		bin = mp_at (spot, bins, i);
		val = img.at<uchar>(bin->y,bin->x);
		if (val != 255)
			continue;
		if (cell_contour (img,drawed,bin->y,bin->x,ctr,ctc,out) != 0)
			continue;

		fprintf (idx_out, "%ld\t%ld\t%d\n", cnt+1, cnt+ctr->n, bin->tidx+1);
		cnt += ctr->n;

		if (++n_plot % 1000 == 0)
			fprintf (stderr, "%d cells have been addressed!\n", n_plot);
	}
	fclose (out);
	fclose (idx_out);

	sprintf (file, "%s.cell_contour.txt", prefix);
	cut_tissue_contour (file);

	sprintf (file, "%s.vec_img.plot.R", prefix);
	out = ckopen (file, "w");

	fprintf (out, "cols <- c('%s'", col_str[0].s);
	for (i=1; i<tnames->n; ++i)
		fprintf (out, ", '%s'", col_str[i].s);
	fprintf (out, ")\n\n");

	fprintf (out, "pdf ('%s.cell_bin.pdf')\n", prefix);
	fprintf (out, "l <- %d\n", img.rows>img.cols ? img.rows : img.cols);
	fprintf (out, "w <- 1.5*l\n");
	fprintf (out, "h <- %.3f * w\n", hw_ratio);
	fprintf (out, "plot (c(1,w), c(1,h), cex=0, xlab='', ylab='', main='', axes=F)\n");
	fprintf (out, "polygon(c(0,0,w,w),c(0,h,h,0), col='%s', border=NA)\n", bg);
	fprintf (out, "ct  <- read.table('%s.cell_contour.txt', header=F, sep='\\t')\n", prefix);
	fprintf (out, "idx <- read.table('%s.cell_contour.idx.txt', header=F, sep='\\t')\n", prefix);
	fprintf (out, "ncell <- nrow(idx)\n");
	fprintf (out, "for (i in 1:ncell) {\n");
	fprintf (out, " y <- ct$V1[idx$V1[i]:idx$V2[i]]\n");
	fprintf (out, " x <- ct$V2[idx$V1[i]:idx$V2[i]]\n");
	fprintf (out, " y <- h - y\n");
	fprintf (out, " x <- x + 0.1*w\n");
	fprintf (out, " y <- y - 0.1*h\n");
	fprintf (out, " polygon (x, y, col=cols[idx$V3[i]], border=NA)\n");
	fprintf (out, "}\n\n");

	int white_line = 1;
	sscanf (bg+1, "%x", &n);
	val = (n&0xff) + ((n>>8)&0xff) + ((n>>16)&0xff);
	if (val > 384)
		white_line = 0;

	fprintf (out, "bar_right <- 0.8 * w\n");
	fprintf (out, "bar_left <- bar_right - %d\n", l_bar*2);
	fprintf (out, "lines (c(bar_left,bar_left),   c(0.05*h,0.05*h+40), col='%s', lwd=0.8)\n", white_line?"#FFFFFF":"#000000");
	fprintf (out, "lines (c(bar_right,bar_right), c(0.05*h,0.05*h+40), col='%s', lwd=0.8)\n", white_line?"#FFFFFF":"#000000");
	fprintf (out, "lines (c(bar_left,bar_right),  c(0.05*h,0.05*h),    col='%s', lwd=0.8)\n", white_line?"#FFFFFF":"#000000");
	fprintf (out, "text ((bar_left+bar_right)/2, 0.05*h+400, '%dum', col='%s', adj=0.5, cex=0.3)\n",
			l_bar, white_line?"#FFFFFF":"#000000");

	fprintf (out, "dev.off()\n");
	fclose (out);

	sprintf (line, "Rscript %s.vec_img.plot.R", prefix);
	system (line);

	printf ("Total plot %d cells\n", n_plot);

	return 0;
}

static int
create_gene_info_from_rds (const char * prefix)
{
	char file[4096];
	char line[4096];
	time_t time_beg;
	FILE * out;

	time (&time_beg);
	fprintf (stderr, "Extract matrix from RDS ...\n");
	sprintf (file, "%s.mtx.extr.R", prefix);
	sprintf (expr_file, "%s.expr_file.txt", prefix);
	out = ckopen (file, "w");
	fprintf (out, "library (Seurat)\n");
	fprintf (out, "library (reshape2)\n\n");
	fprintf (out, "obj <- readRDS ('%s')\n", rds_file);
	fprintf (out, "gene <- read.table('%s')\n", gene_list_file);
	if (run_norm) {
		fprintf (out, "obj <- NormalizeData (obj)\n");
	}

	fprintf (out, "coor <- obj@images$slice1@coordinates\n");
	fprintf (out, "coor <- coor[,c(3,2)]\n");
	fprintf (out, "coor$cell <- rownames(coor)\n");

	if (data_type == 0)
		fprintf (out, "data <- as.data.frame(obj@assays$Spatial@counts)[gene$V1,]\n");
	else
		fprintf (out, "data <- as.data.frame(obj@assays$Spatial@data)[gene$V1,]\n");

	fprintf (out, "data1 <- cbind(row = rownames(data), melt(data))\n");

	fprintf (out, "rm (obj)\n");
	fprintf (out, "rm (data)\n");
	fprintf (out, "gc ()\n");

	fprintf (out, "data1 <- na.omit (data1)\n");
	fprintf (out, "all_cell <- unique (data1$variable)\n");
	fprintf (out, "non_zero_data <- subset (data1, value>0)\n");
	fprintf (out, "non_zero_cell <- unique (non_zero_data$variable)\n");
	fprintf (out, "zero_cell <- setdiff (all_cell, non_zero_cell)\n");
	fprintf (out, "zero_data <- data.frame (row=data1[1,1], variable=zero_cell, value=0)\n");
	fprintf (out, "expr <- rbind (non_zero_data, zero_data)\n");
	fprintf (out, "expr <- merge (expr, coor, by.x='variable', by.y='cell')\n");
	fprintf (out, "expr <- expr[c(4,5,2,3)]\n");

	fprintf (out, "write.table (expr, file='%s', row.names=F, col.names=F, sep='\\t', quote=F)\n\n", expr_file);
	fclose (out);

	sprintf (line, "Rscript %s.mtx.extr.R", prefix);
	system (line);
	fprintf (stderr, "Extract expression from RDS cost %lds\n", time(NULL)-time_beg);

	/*
	time (&time_beg);
	fprintf (stderr, "Prepare data for plot ...\n");
	sprintf (file, "%s.prep.pl", prefix);
	out = ckopen (file, "w");
	fprintf (out, "#!/usr/bin/perl -w\n");
	fprintf (out, "use strict;\n\n");
	fprintf (out, "my %%gene = ();\n");
	fprintf (out, "my @t = ();\n");
	fprintf (out, "open IN, \"%s\" or die $!;\n", gene_list_file);
	fprintf (out, "while (<IN>) {\n");
	fprintf (out, "  chomp;\n");
	fprintf (out, "  @t = split;\n");
	fprintf (out, "  $gene{$t[0]} = 1;\n");
	fprintf (out, "}\n");
	fprintf (out, "close IN;\n\n");
	fprintf (out, "my %%cell_pos = ();\n");
	fprintf (out, "my %%cell_mis = ();\n");
	fprintf (out, "open IN, \"%s.metadata.txt\" or die $!;\n", prefix);
	fprintf (out, "my $head = <IN>;\n");
	fprintf (out, "while (<IN>) {\n");
	fprintf (out, "  chomp;\n");
	fprintf (out, "  @t = split;\n");
	fprintf (out, "  my $val = \"$t[-1]\t$t[-2]\";\n");
	fprintf (out, "  $cell_pos{$t[0]} = $val;\n");
	fprintf (out, "  $cell_mis{$t[0]} = $val;\n");
	fprintf (out, "}\n");
	fprintf (out, "close IN;\n\n");
	fprintf (out, "my %%idx2gene = ();\n");
	fprintf (out, "open IN, \"%s.expr_mtx.txt\" or die $!;\n", prefix);
	fprintf (out, "$head = <IN>;\n");
	fprintf (out, "chomp ($head);\n");
	fprintf (out, "@t = split /\\t/, $head;\n");
	fprintf (out, "my $mis_gene = \"\";\n");
	fprintf (out, "foreach my $idx (0..$#t) {\n");
	fprintf (out, "  my $new_idx = $idx + 1;\n");
	fprintf (out, "  next if (!exists($gene{$t[$idx]}));\n");
	fprintf (out, "  $idx2gene{$new_idx} = $t[$idx];\n");
	fprintf (out, "  $mis_gene = $t[$idx];\n");
	fprintf (out, "}\n\n");
	fprintf (out, "open OUT, \"> %s\" or die $!;\n", expr_file);
	fprintf (out, "while (<IN>) {\n");
	fprintf (out, "  chomp;\n");
	fprintf (out, "  @t = split;\n");
	fprintf (out, "  next if (!exists($cell_pos{$t[0]}));\n");
	fprintf (out, "  foreach (1..$#t) {\n");
	fprintf (out, "    next if ($t[$_] == 0);\n");
	fprintf (out, "    next if (!exists($idx2gene{$_}));\n");
	fprintf (out, "    print OUT $cell_pos{$t[0]}, \"\\t\", $idx2gene{$_}, \"\\t\", $t[$_], \"\\n\";\n");
	fprintf (out, "    $cell_mis{$t[0]} = 0;\n");
	fprintf (out, "  }\n");
	fprintf (out, "}\n");
	fprintf (out, "\n");
	fprintf (out, "foreach (keys %%cell_mis) {\n");
	fprintf (out, "  next if (!$cell_mis{$_});\n");
	fprintf (out, "  print OUT $cell_mis{$_}, \"\\t$mis_gene\\t0\\n\";\n");
	fprintf (out, "}\n");
	fprintf (out, "\n");
	fprintf (out, "close IN;\n");
	fprintf (out, "close OUT;\n");
	fclose (out);

	sprintf (line, "perl %s.prep.pl", prefix);
	system (line);
	fprintf (stderr, "Prepare data for plot cost %lds\n", time(NULL)-time_beg);
	*/

	return 0;
}

static int
count_gem_ncols (void)
{
	char line[4096];
	char * ch;
	int n_blk;
	gzFile in = ckgzopen (gem_file, "r");

	while (gzgets(in,line,4096))
		if (*line != '#')
			break;

	for (ch=line,n_blk=0; *ch!='\0'; ++ch)
		if (isspace(*ch))
			++n_blk;

	gzclose (in);

	return n_blk;
}

static int
create_gene_info_from_gem (const char * prefix)
{
	char line[4096];
	char gene[4096];
	int x, y;
	int ncol;
	int * cell_x;
	int * cell_y;
	int * cell_flag;
	time_t time_beg;
	double val;
	double * val_ptr;
	double * expr_mtx;
	int64_t i, j;
	int64_t ngenes;
	int64_t ncells;
	int64_t gene_idx;
	int64_t cell_idx;
	uint64_t cell_label;
	FILE * in;
	FILE * out;
	gzFile gem_in;
	str_t s;
	str_t * str_ptr;
	str_set_t * genes;
	xh_item_t * xh_ptr;
	str_hash_t * gene_hash;
	xh_set_t(u64) * cell_hash;

	time (&time_beg);
	fprintf (stderr, "Begin creating gene info from GEM ...\n");

	gene_hash = str_hash_init ();
	cell_hash = xh_u64_set_init ();

	in = ckopen (gene_list_file, "r");
	while (fgets(line,4096,in)) {
		sscanf (line, "%s", gene);
		s.s = gene;
		s.l = strlen (gene);
		str_hash_add (gene_hash, &s);
	}
	fclose (in);

	ncol = count_gem_ncols ();
	if (ncol < 5)
		err_mesg ("command 'gene' only accept cellbin gem file with cell label column");

	gem_in = ckgzopen (gem_file, "r");
	gzgets (gem_in, line, 4096);
	while (gzgets(gem_in,line,4096)) {
		sscanf (line, "%*s %*s %*s %*s %lu", &cell_label);
		xh_set_add (u64, cell_hash, &cell_label);
	}

	ngenes = xh_set_cnt (gene_hash);
	ncells = xh_set_cnt (cell_hash);
	printf ("Total %ld genes\n", ngenes);
	printf ("Total %ld cells\n", ncells);

	cell_x    = (int *) ckmalloc (ncells * sizeof(int));;
	cell_y    = (int *) ckmalloc (ncells * sizeof(int));;
	expr_mtx  = (double *) ckalloc (ngenes*ncells, sizeof(double));
	cell_flag = (int *) ckalloc (ncells, sizeof(int));

	genes = str_set_init ();
	xh_set_key_iter_init (xstr, gene_hash);
	while ((str_ptr = xh_set_key_iter_next(xstr,gene_hash)) != NULL)
		str_set_add2 (genes, str_ptr->s, str_ptr->l);

	for (i=0; i<ncells; ++i)
		cell_x[i] = -1;
	gzclose (gem_in);

	gem_in = ckgzopen (gem_file, "r");
	gzgets (gem_in, line, 4096);
	while (gzgets(gem_in,line,4096)) {
		sscanf (line, "%s %d %d %lf %lu", gene, &x, &y, &val, &cell_label);

		xh_ptr = xh_set_search3 (u64, cell_hash, &cell_label);
		cell_idx = xh_ptr->id;

		if (cell_x[cell_idx] < 0) {
			cell_x[cell_idx] = x;
			cell_y[cell_idx] = y;
		}

		s.s = gene;
		s.l = strlen (gene);
		if ((xh_ptr=str_hash_search3(gene_hash,&s)) == NULL)
			continue;
		gene_idx = xh_ptr->id;

		val_ptr = expr_mtx + cell_idx*ngenes + gene_idx;
		*val_ptr += val;
	}
	gzclose (gem_in);

	sprintf (expr_file, "%s.expr_file.txt", prefix);
	out = ckopen (expr_file, "w");
	for (i=0; i<ncells; ++i)
		for (j=0; j<ngenes; ++j) {
			val = expr_mtx[i*ngenes+j];
			if (val < 1e-10)
				continue;
			fprintf (out, "%d\t%d\t%s\t%e\n",
					cell_x[i], cell_y[i],
					genes->pool[j].s, val);
			cell_flag[i] = 1;
		}

	for (i=0; i<ncells; ++i)
		if (!cell_flag[i])
			fprintf (out, "%d\t%d\t%s\t0\n",
					cell_x[i], cell_y[i],
					genes->pool[0].s);
	fclose (out);

	fprintf (stderr, "Finish creating gene info from GEM, cost %lds\n\n", time(NULL)-time_beg);

	return 0;
}

static void
heatmap_rgb (FILE * out)
{
	if (*color_file) {
		fprintf (out, "cols <- read.table('%s', comment.char='')$V1\n", color_file);
		fprintf (out, "pal <- colorRampPalette (cols)(1000)\n");
		return;
	}

	fprintf (out, "rgb2str <- function (r, g, b) {\n");
	fprintf (out, "  str <- sprintf (\"#%%02x%%02x%%02x\", r, g, b)\n");
	fprintf (out, "  return (str)\n");
	fprintf (out, "}\n");
	fprintf (out, "\n");
	fprintf (out, "rgbs2strs <- function (r, g, b) {");
	fprintf (out, "  lr <- length (r)\n");
	fprintf (out, "  lg <- length (g)\n");
	fprintf (out, "  lb <- length (b)\n");
	fprintf (out, "  if (lr!=lg | lr!=lb | lg!=lb) {\n");
	fprintf (out, "    stop (\"length of r, g, b must be same!\");\n");
	fprintf (out, "  }\n");
	fprintf (out, "\n");
	fprintf (out, "  strs <- rep (\"#000000\", lr)\n");
	fprintf (out, "  for (i in 1:lr) {\n");
	fprintf (out, "    strs[i] <- rgb2str (r[i],g[i],b[i])\n");
	fprintf (out, "  }\n");
	fprintf (out, "\n");
	fprintf (out, "  return (strs)\n");
	fprintf (out, "}\n");
	fprintf (out, "\n");
	fprintf (out, "b <- c(131, 163, 186, 153, 56, 50, 56, 36, 30)\n");
	fprintf (out, "g <- c(51, 94, 136, 193, 211, 177, 143, 97, 30)\n");
	fprintf (out, "r <- c(12, 0, 10, 0, 242, 246, 242, 231, 217)\n");
	fprintf (out, "cols <- rgbs2strs (r, g, b);\n");
	fprintf (out, "pal <- colorRampPalette (cols)(1000)\n");
}

static void
output_background_cell_contours (const char * prefix, Mat & img, Mat & drawed, arr_t(i32) * ctr, arr_t(i32) * ctc)
{
	char file[4096];
	char l[4096];
	int val;
	int row, col;
	int n_plot;
	int debug;
	int64_t cnt;
	FILE * in;
	FILE * out;
	FILE * idx_out;

	sprintf (file, "%s.all_cell_contour.txt", prefix);
	out = ckopen (file, "w");
	sprintf (file, "%s.all_cell_contour.idx.txt", prefix);
	idx_out = ckopen (file, "w");

	cnt = 0;
	n_plot = 0;
	in = ckopen (all_cell_pfile, "r");
	while (fgets(l,4096,in)) {
		sscanf (l, "%d %d", &col, &row);
		val = img.at<uchar>(row,col);
		if (val == 0)
			continue;

		if (row==119 && col==108)
			debug = 1;

		if (cell_contour (img,drawed,row,col,ctr,ctc,out) != 0) {
			printf ("%d\t%d\n", row, col);
			continue;
		}

		fprintf (idx_out, "%ld\t%ld\n", cnt+1, cnt+ctr->n);
		cnt += ctr->n;

		if (++n_plot % 1000 == 0)
			fprintf (stderr, "%d backgroud cells have been processed!\n", n_plot);
	}
	fclose (in);
	fclose (out);
	fclose (idx_out);
}

static int
expr_plot (const char * prefix)
{
	char buff[4096];
	char line[4096];
	char file[4096];
	char * ch;
	int32_t n_plot;
	int32_t x, y;
	int32_t i, j;
	int32_t ncells;
	int32_t ngenes;
	int32_t cell_idx;
	int32_t gene_idx;
	float val;
	float * val_ptr;
	float * expr_mtx;
	time_t time_beg;
	int64_t cnt;
	uint64_t key;
	uint64_t * u64_ptr;
	FILE * in;
	FILE * out;
	FILE * idx_out;
	Mat img;
	Mat drawed;
	str_t s;
	str_t * gene;
	str_t * str_ptr;
	bin_t * bin;
	bin_t * cells;
	xh_item_t * xh_ptr;
	str_t * genes;
	str_hash_t * gene_hash;
	xh_set_t(u64) * cell_hash;
	arr_t(i32) * ctr;
	arr_t(i32) * ctc;

	time (&time_beg);
	fprintf (stderr, "Begin drawing cell contour and plot script ...\n");

	gene_hash = str_hash_init ();
	cell_hash = xh_u64_set_init ();

	in = ckopen (expr_file, "r");
	while (fgets(line,4096,in)) {
		sscanf (line, "%d %d %s", &x, &y, buff);
		key = ((uint64_t)x << 32) | (uint64_t)y;
		xh_set_add (u64, cell_hash, &key);

		for (ch=buff; *ch!='\0'; ++ch) {
			if (*ch=='-' || *ch==',' || *ch==';')
				*ch = '_';
		}

		s.s = buff;
		s.l = strlen (buff);
		str_hash_add (gene_hash, &s);
	}
	ncells = xh_set_cnt (cell_hash);
	ngenes = xh_set_cnt (gene_hash);
	printf ("%d cells in expression file\n", ncells);
	printf ("%d genes in expression file\n", ngenes);

	expr_mtx = (float *) ckalloc (ncells*ngenes, sizeof(float));

	rewind (in);
	while (fgets(line,4096,in)) {
		sscanf (line, "%d %d %s %f", &x, &y, buff, &val);
		key = ((uint64_t)x << 32) | (uint64_t)y;

		xh_ptr = xh_set_search3 (u64, cell_hash, &key);
		cell_idx = xh_ptr->id;

		for (ch=buff; *ch!='\0'; ++ch) {
			if (*ch=='-' || *ch==',' || *ch==';')
				*ch = '_';
		}

		s.s = buff;
		s.l = strlen (buff);
		xh_ptr = str_hash_search3 (gene_hash, &s);
		gene_idx = xh_ptr->id;

		*(expr_mtx + cell_idx*ngenes + gene_idx) = val;
	}
	fclose (in);

	xcv_imread_gray (img, mask_img);
	drawed = Mat::zeros (img.size(), CV_8UC1);
	for (i=0; i<img.rows; ++i)
		for (j=0; j<img.cols; ++j)
			if (img.at<uchar>(i,j) > 0)
				img.at<uchar>(i,j) = 255;

	i = 0;
	int tmp;
	cells = (bin_t *) ckalloc (ncells, sizeof(bin_t));
	xh_set_key_iter_init (u64, cell_hash);
	while ((u64_ptr = xh_set_key_iter_next(u64,cell_hash)) != NULL) {
		key = *u64_ptr;
		y = (int32_t) (key & 0xffffffff);
		x = (int32_t) ((key>>32) & 0xffffffff);

		x = x - 1;
		y = y - 1;
		if (x < 0) x = 0;
		if (y < 0) y = 0;

		if (ex_xy) {
			tmp = x;
			x = y;
			y = tmp;
		}

		if (x >= img.cols || y >= img.rows)
			continue;

		bin = cells + i++;
		bin->x = x;
		bin->y = y;
	}
	printf ("%d\n", ncells);

	i = 0;
	genes = (str_t *) ckalloc (ngenes, sizeof(str_t));
	xh_set_key_iter_init (xstr, gene_hash);
	while ((str_ptr = xh_set_key_iter_next(xstr,gene_hash)) != NULL) {
		gene = genes + i++;
		str_copy (gene, str_ptr);
	}

	cnt = 0;
	n_plot = 0;
	ctr = arr_init (i32);
	ctc = arr_init (i32);
	sprintf (file, "%s.cell_contour.txt", prefix);
	out = ckopen (file, "w");
	sprintf (file, "%s.cell_contour.idx.txt", prefix);
	idx_out = ckopen (file, "w");

	fprintf (idx_out, "start\tend");
	for (j=0; j<ngenes; ++j)
		fprintf (idx_out, "\t%s", genes[j].s);
	fprintf (idx_out, "\n");

	for (i=0; i<ncells; ++i) {
		bin = cells + i;
		val = img.at<uchar>(bin->y,bin->x);
		if (val == 0)
			continue;
		if (cell_contour (img,drawed,bin->y,bin->x,ctr,ctc,out) != 0)
			continue;

		fprintf (idx_out, "%ld\t%ld", cnt+1, cnt+ctr->n);
		cnt += ctr->n;

		val_ptr = expr_mtx + i*ngenes;
		for (j=0; j<ngenes; ++j)
			fprintf (idx_out, "\t%.3f", val_ptr[j]);
		fprintf (idx_out, "\n");

		if (++n_plot % 1000 == 0)
			fprintf (stderr, "%d cells have been processed!\n", n_plot);
	}
	fclose (out);
	fclose (idx_out);

	if (*all_cell_pfile) {
		drawed = Mat::zeros (img.size(), CV_8UC1);
		output_background_cell_contours (prefix, img, drawed, ctr, ctc);
	}

	sprintf (file, "%s.cell_contour.txt", prefix);
	cut_tissue_contour (file);

	sprintf (file, "%s.vec_img.plot.R", prefix);
	out = ckopen (file, "w");

	heatmap_rgb (out);
	fprintf (out, "\n");
	fprintf (out, "pdf ('%s.cell_bin.pdf')\n", prefix);
	fprintf (out, "l <- %d\n", img.rows>img.cols ? img.rows : img.cols);
	fprintf (out, "w <- 1.5 * l\n");
	fprintf (out, "h <- %.3f * w\n", hw_ratio);
	fprintf (out, "\n");
	fprintf (out, "ct <- read.table('%s.cell_contour.txt', header=F, sep='\\t')\n", prefix);
	fprintf (out, "expr <- read.table('%s.cell_contour.idx.txt', header=T, sep='\\t', check.names=F)\n", prefix);
	fprintf (out, "ncell <- nrow (expr)\n");
	fprintf (out, "ngene <- ncol (expr) - 2\n");
	fprintf (out, "start <- expr$start\n");
	fprintf (out, "end <- expr$end\n");
	fprintf (out, "header <- colnames(expr)\n");
	fprintf (out, "\n");

	if (*all_cell_pfile) {
	fprintf (out, "all_cell_ct <- read.table('%s.all_cell_contour.txt', header=F, sep='\\t')\n", prefix);
	fprintf (out, "all_cell_ct_idx <- read.table('%s.all_cell_contour.idx.txt', header=F, sep='\\t')\n", prefix);
	fprintf (out, "all_cell_start <- all_cell_ct_idx$V1\n");
	fprintf (out, "all_cell_end <- all_cell_ct_idx$V2\n");
	fprintf (out, "n_all_cell <- nrow (all_cell_ct_idx)\n");
	}

	for (i=0; i<ngenes; ++i) {
		fprintf (out, "### Plot Gene %s\n", genes[i].s);
		fprintf (out, "print ('Plot Gene %s')\n", genes[i].s);
		fprintf (out, "val <- expr[['%s']]\n", genes[i].s);
		if (data_type == 1)
			fprintf (out, "max_val = max (val)\n");
		else
			fprintf (out, "max_val = quantile (val, 0.99)\n");
		fprintf (out, "if (max_val > 0.1) {\n");
		if (max_expr > 0)
			fprintf (out, "  if (max_val > %.3f) {max_val = %.3lf}\n\n", max_expr, max_expr);
		fprintf (out, "  plot (c(1,w), c(1,h), cex=0, xlab='', ylab='', main='', axes=F)\n");
		fprintf (out, "  polygon(c(0,0,w,w),c(0,h,h,0), col='%s', border=NA)\n", bg);
		fprintf (out, "  text (0.5*w, 0.9*h, expression(italic(\"%s\")), adj=0.5, col='%s')\n", genes[i].s, text_color);
		fprintf (out, "\n");
		fprintf (out, "  for (j in 1:ncell) {\n");
		fprintf (out, "    y <- ct$V1[start[j]:end[j]]\n");
		fprintf (out, "    x <- ct$V2[start[j]:end[j]]\n");
		fprintf (out, "    y <- h - y\n");
		fprintf (out, "    x <- x + 0.05*w\n");
		fprintf (out, "    y <- y - 0.2*h\n");
		fprintf (out, "    col_idx <- as.integer (val[j]*1000/max_val) + 1\n");
		fprintf (out, "    if (col_idx<1) {col_idx=1}\n");
		fprintf (out, "    if (col_idx>1000) {col_idx=1000}\n");
		fprintf (out, "    polygon (x, y, col=pal[col_idx], border=NA)\n");
		fprintf (out, "  }\n");

		if (*all_cell_pfile) {
		fprintf (out, "  for (j in 1:n_all_cell) {\n");
		fprintf (out, "    y <- all_cell_ct$V1[all_cell_start[j]:all_cell_end[j]]\n");
		fprintf (out, "    x <- all_cell_ct$V2[all_cell_start[j]:all_cell_end[j]]\n");
		fprintf (out, "    y <- h - y\n");
		fprintf (out, "    x <- x + 0.05*w\n");
		fprintf (out, "    y <- y - 0.2*h\n");
		fprintf (out, "    polygon (x, y, col='#DDDDDD', border=NA)\n");
		fprintf (out, "  }\n");
		}

		fprintf (out, "\n");
		fprintf (out, "  ly <- seq (h/3, 2*h/3, length.out=1001)\n");
		fprintf (out, "  for (i in 1:1000) {\n");
		fprintf (out, "    lxb <- 0.8*w\n");
		fprintf (out, "    lxe <- 0.8*w + 100\n");
		fprintf (out, "    rect (xleft=lxb, xright=lxe, ybottom=ly[i], ytop=ly[i+1], border=NA, col=pal[i])\n");
		fprintf (out, "  }\n");
		fprintf (out, "  text (0.8*w+150, ly[1], \"0\", adj=0, col='%s')\n", text_color);
		fprintf (out, "  ls <- sprintf (\"%%.1f\", max_val)\n");
		fprintf (out, "  text (0.8*w+150, ly[1000], ls, adj=0, col='%s')\n", text_color);
		fprintf (out, "}\n");
		fprintf (out, "\n");
	}

	fprintf (out, "dev.off()\n");
	fclose (out);

	fprintf (stderr, "Finish drawing cell contour and plot script, cost %lds\n\n", time(NULL)-time_beg);

	time (&time_beg);
	fprintf (stderr, "Begin running plot script ...\n");
	sprintf (line, "Rscript %s.vec_img.plot.R", prefix);
	system (line);
	fprintf (stderr, "Finish running plot script, cost %lds\n\n", time(NULL)-time_beg);

	return 0;
}

// cluster plot

int
cluster_main (int argc, char * argv[])
{
	if (argc < 2) {
		fprintf (stderr, "\n");
		fprintf (stderr, "Usage:   cellbin_vec_plot cluster [options] <mask.tif>\n");
		fprintf (stderr, "Contact: chenxi1@genomics.cn\n");
		fprintf (stderr, "Options: -i   STR   Cell cluster info file for each cell\n");
		fprintf (stderr, "         -c   STR   List of colors for clusters\n");
		fprintf (stderr, "         -o   STR   Output directory\n");
		fprintf (stderr, "         -s   STR   Output sample name\n");
		fprintf (stderr, "         -l   INT   Length of bar in unit um (default: 500)\n");
		fprintf (stderr, "         -b   STR   Backgroud for image (default: #000000)\n");
		fprintf (stderr, "         -r   FLT   Ratio of canvas height/width (default: 1.0)\n");
		fprintf (stderr, "\n");
		return 1;
	}

	char prefix[4096];
	int copt;
	time_t time_beg;

	while ((copt=getopt(argc,argv,"i:c:o:s:l:b:r:")) != -1) {
		if (copt == 'i')
			strcpy (cluster_file, optarg);
		else if (copt == 'c')
			strcpy (color_file, optarg);
		else if (copt == 'o')
			strcpy (out_dir, optarg);
		else if (copt == 's')
			strcpy (spl_name, optarg);
		else if (copt == 'l')
			l_bar = atoi (optarg);
		else if (copt == 'b')
			strcpy (bg, optarg);
		else if (copt == 'r')
			hw_ratio = atoi (optarg);
	}
	if (optind >= argc)
		err_mesg ("fail to get cell mask image!");
	else
		mask_img = strdup (argv[optind++]);

	if (l_bar <= 0)
		err_mesg ("bar length must > 0");

	if (!*cluster_file || !*color_file)
		err_mesg ("color file and cluster file are required!");

	if (hw_ratio <= 0)
		err_mesg ("Ratio of canvas height/width must > 0");

	if (!*out_dir)
		strcpy (out_dir, ".");
	if (!*spl_name)
		strcpy (spl_name, "sample");

	if (!*bg)
		strcpy (bg, "#000000");
	set_text_color ();

	time (&time_beg);

	sprintf (prefix, "%s/%s", out_dir, spl_name);

	cellbin_cluster_plot (prefix);

	printf ("Program cost %lds\n", time(NULL)-time_beg);

	return 0;
}


// gene plot

int
gene_main (int argc, char * argv[])
{
	if (argc < 2) {
		fprintf (stderr, "\n");
		fprintf (stderr, "Usage:   cellbin_vec_plot gene [options] <mask.tif>\n");
		fprintf (stderr, "Contact: chenxi1@genomics.cn\n");
		fprintf (stderr, "Options: -v   STR   File containing expression values for cells\n");
		fprintf (stderr, "         -r   STR   RDS file of cell bin data\n");
		fprintf (stderr, "         -e   STR   GEM file of cell bin data\n");
		fprintf (stderr, "         -g   STR   Gene list for plot, required when '-r' or '-e' is provided\n");
		fprintf (stderr, "         -d   INT   Type of data extracted from RDS (default: 1)\n");
		fprintf (stderr, "                    0 for raw counts,\n");
		fprintf (stderr, "                    1 for normalized counts\n");
		fprintf (stderr, "         -m   INT   Maximum expression value for plot\n");
		fprintf (stderr, "                    defaut: when using normalized data, not used;\n");
		fprintf (stderr, "                            when using raw data, 99%% quantile\n");
		fprintf (stderr, "         -o   STR   Output directory\n");
		fprintf (stderr, "         -s   STR   Output sample name\n");
		fprintf (stderr, "         -l   INT   Length of bar in unit um (default: 500)\n");
		fprintf (stderr, "         -c   STR   List of colors for heatmap\n");
		fprintf (stderr, "         -b   STR   Backgroud for image (default: #000000)\n");
		fprintf (stderr, "         -x         exchange row<->column index in input file\n");
		fprintf (stderr, "         -n         Run NormalizeData for raw RDS\n");
		fprintf (stderr, "         -B   STR   File containing x/y of all cells, used as gray backgroud\n");
		fprintf (stderr, "\n");
		return 1;
	}

	char prefix[4096];
	int copt;
	time_t time_beg;

	while ((copt=getopt(argc,argv,"v:r:e:g:d:m:o:s:l:c:b:xnB:")) != -1) {
		if (copt == 'v')
			strcpy (expr_file, optarg);
		else if (copt == 'r')
			strcpy (rds_file, optarg);
		else if (copt == 'e')
			strcpy (gem_file, optarg);
		else if (copt == 'g')
			strcpy (gene_list_file, optarg);
		else if (copt == 'd') {
			data_type = atoi (optarg);
			if (data_type!=0 && data_type!=1)
				err_mesg ("parameter '-d' must be set to 0 or 1");
		} else if (copt == 'm')
			max_expr = (float) atof (optarg);
		else if (copt == 'o')
			strcpy (out_dir, optarg);
		else if (copt == 's')
			strcpy (spl_name, optarg);
		else if (copt == 'l')
			l_bar = atoi (optarg);
		else if (copt == 'c')
			strcpy (color_file, optarg);
		else if (copt == 'b')
			strcpy (bg, optarg);
		else if (copt == 'x')
			ex_xy = 1;
		else if (copt == 'n')
			run_norm = 1;
		else if (copt == 'B')
			strcpy (all_cell_pfile, optarg);
	}
	if (optind >= argc)
		err_mesg ("fail to get cell mask image!");
	else
		mask_img = strdup (argv[optind++]);

	if (l_bar <= 0)
		err_mesg ("bar length must > 0");

	if (!*out_dir)
		strcpy (out_dir, ".");
	if (!*spl_name)
		strcpy (spl_name, "sample");

	if (!*bg)
		strcpy (bg, "#000000");
	set_text_color ();

	time (&time_beg);

	sprintf (prefix, "%s/%s", out_dir, spl_name);

	if (*rds_file) { // RDS file
		if (!*gene_list_file)
			err_mesg ("'-g' must be privided, when '-r' is set");
		create_gene_info_from_rds (prefix);
		expr_plot (prefix);
	} else if (*gem_file) { // GEM file
		if (!*gene_list_file)
			err_mesg ("'-g' must be privided, when '-e' is set");
		data_type = 0;
		create_gene_info_from_gem (prefix);
		expr_plot (prefix);
	} else if (*expr_file) { // Expression Plot
		expr_plot (prefix);
	}

	printf ("Program cost %lds\n", time(NULL)-time_beg);

	return 0;
}

// map plot

typedef struct {
	int row;
	int col;
	int blk;
	int spatial_cluster;
	double * score;
} cell_t;

static int
map_main (int argc, char * argv[])
{
	if (argc < 2) {
		fprintf (stderr, "\n");
		fprintf (stderr, "Usage:   cellbin_vec_plot map [options] <mask.tif>\n");
		fprintf (stderr, "Contact: chenxi1@genomics.cn\n");
		fprintf (stderr, "Options: -m   STR   File containing prediction score matrix for cells\n");
		fprintf (stderr, "         -o   STR   Output directory\n");
		fprintf (stderr, "         -s   STR   Output sample name\n");
		fprintf (stderr, "         -M   FLT   Minimum value for max score in one cluster (default: 0.3)\n");
		fprintf (stderr, "         -b   STR   Backgroud for image (default: #000000)\n");
		fprintf (stderr, "\n");

		return 1;
	}

	char file[4096];
	char line[4096];
	char * ch;
	char * pred_file;
	int i, j;
	int row;
	int col;
	int copt;
	int ncells;
	int ncls;
	time_t time_beg;
	double min_maxval;
	FILE * in;
	FILE * out;
	FILE * idx_out;
	cell_t * cell;
	cell_t * cells;

	min_maxval = 0.3;
	pred_file = ALLOC_LINE;
	while ((copt = getopt(argc,argv,"m:o:s:M:b:")) != -1) {
		if (copt == 'm')
			strcpy (pred_file, optarg);
		else if (copt == 'o')
			strcpy (out_dir, optarg);
		else if (copt == 's')
			strcpy (spl_name, optarg);
		else if (copt == 'M')
			min_maxval = atof (optarg);
		else if (copt == 'b')
			strcpy (bg, optarg);
	}

	if (!*out_dir)
		strcpy (out_dir, ".");
	if (!*spl_name)
		strcpy (spl_name, "sample");

	if (optind >= argc)
		err_mesg ("fail to get mask image file!");
	else
		mask_img = strdup (argv[optind++]);

	if (!*bg)
		strcpy (bg, "#000000");
	set_text_color ();

	time (&time_beg);

	sprintf (file, "%s/%s.format.R", out_dir, spl_name);
	out = ckopen (file, "w");
	fprintf (out, "rt <- read.table (\"%s\", header=T, check.names=F)\n", pred_file);
	fprintf (out, "cluster <- as.integer(colnames(rt))\n");
	fprintf (out, "cluster <- cluster[!is.na(cluster)]\n");
	fprintf (out, "cluster <- as.character (sort (cluster))\n");
	fprintf (out, "nrows = nrow (rt)\n");
	fprintf (out, "ncols = length (cluster)\n");
	fprintf (out, "mtx <- matrix (0, nrow=nrows, ncol=ncols+3)\n");
	fprintf (out, "mtx[,1] <- rt$imagerow\n");
	fprintf (out, "mtx[,2] <- rt$imagecol\n");
	fprintf (out, "mtx[,3] <- rt$spatial_cls\n");
	fprintf (out, "for (i in 1:ncols) { \n");
	fprintf (out, "	cls <- cluster[i]\n");
	fprintf (out, "	val <- rt[[cls]]\n");
	fprintf (out, "	mtx[,i+3] = val\n");
	fprintf (out, "}\n");
	fprintf (out, "write.table (mtx, file='%s/%s.fmt.txt', quote=F, sep='\\t', row.names=F, col.names=F)\n", out_dir, spl_name);
	fclose (out);

	sprintf (line, "Rscript %s/%s.format.R", out_dir, spl_name);
	system (line);

	sprintf (file, "%s/%s.fmt.txt", out_dir, spl_name);
	in = ckopen (file, "r");
	ncells = 0;
	while (fgets(line,4096,in))
		++ncells;

	rewind (in);
	cells = (cell_t *) ckalloc (ncells, sizeof(cell_t));
	ncells = 0;

	fgets (line, 4096, in);
	chomp (line);
	strcpy (file, line);

	assert ((ch = strtok(line,"\t ")) != NULL);
	assert ((ch = strtok(NULL,"\t ")) != NULL);
	assert ((ch = strtok(NULL,"\t ")) != NULL);
	ncls = 0;
	while (ch = strtok(NULL,"\t "))
		++ncls;

	cell = cells + ncells++;
	cell->score = (double *) ckalloc (ncls, sizeof(double));

	strcpy (line, file);
	assert ((ch = strtok(line,"\t ")) != NULL);
	cell->row = stoi (ch);
	assert ((ch = strtok(NULL,"\t ")) != NULL);
	cell->col = stoi (ch);
	assert ((ch = strtok(NULL,"\t ")) != NULL);
	cell->spatial_cluster = atoi (ch);
	i = 0;
	while (ch = strtok(NULL,"\t "))
		cell->score[i++] = atof (ch);

	while (fgets(line,4096,in)) {
		cell = cells + ncells++;
		cell->score = (double *) ckalloc (ncls, sizeof(double));
		chomp (line);

		assert ((ch = strtok(line,"\t ")) != NULL);
		cell->row = stoi (ch);
		assert ((ch = strtok(NULL,"\t ")) != NULL);
		cell->col = stoi (ch);
		assert ((ch = strtok(NULL,"\t ")) != NULL);
		cell->spatial_cluster = atoi (ch);
		i = 0;
		while (ch = strtok(NULL,"\t "))
			cell->score[i++] = atof (ch);
	}
	fclose (in);

	int row_max = -1;
	int col_max = -1;
	for (i=0; i<ncells; ++i) {
		cell = cells + i;
		if (cell->row > row_max) row_max = cell->row;
		if (cell->col > col_max) col_max = cell->col;
	}
	int val;
	int64_t cnt = 0;
	int64_t n_plot = 0;
	arr_t(i32) * ctr = arr_init (i32);
	arr_t(i32) * ctc = arr_init (i32);
	Mat mask;
	Mat drawed;
	int interchange = 0;
	double ratio;
	double ratio2;

	xcv_imread_gray (mask, mask_img);
	ratio = (double)row_max*mask.cols/col_max/mask.rows;
	ratio2 = (double)row_max*mask.rows/col_max/mask.cols;
	if (fabs(1-ratio) > fabs(1-ratio2))
		interchange = 1;

	for (i=0; i<mask.rows; ++i)
		for (j=0; j<mask.cols; ++j)
			if (mask.at<uchar>(i,j) > 0)
				mask.at<uchar>(i,j) = 255;

	drawed = Mat::zeros (mask.size(), CV_8UC1);

	sprintf (file, "%s/%s.scRNA2spatial.cell_contour.txt", out_dir, spl_name);
	out = ckopen (file, "w");
	sprintf (file, "%s/%s.scRNA2spatial.cell_contour.idx.txt", out_dir, spl_name);
	idx_out = ckopen (file, "w");
	for (i=0; i<ncells; ++i) {
		cell = cells + i;
		if (interchange) {
			row = cell->col;
			col = cell->row;
		} else {
			row = cell->row;
			col = cell->col;
		}
		val = mask.at<uchar>(row,col);
		if (val != 255)
			continue;
		if (cell_contour (mask,drawed,row,col,ctr,ctc,out) != 0)
			continue;

		fprintf (idx_out, "%ld\t%ld", cnt+1, cnt+ctr->n);
		for (j=0; j<ncls; ++j)
			fprintf (idx_out, "\t%.6lf", cell->score[j]);
		fprintf (idx_out, "\n");

		cnt += ctr->n;

		if (++n_plot % 1000 == 0)
			fprintf (stderr, "%ld cells have been addressed!\n", n_plot);
	}
	fclose (out);
	fclose (idx_out);
	fprintf (stderr, "Total %d cells, plot %ld cells\n", ncells, n_plot);

	sprintf (file, "%s/%s.scRNA2spatial.vec_img.plot.R", out_dir, spl_name);
	out = ckopen (file, "w");

	heatmap_rgb (out);
	fprintf (out, "\n");
	fprintf (out, "pdf ('%s/%s.scRNA2spatial.cell_bin.pdf')\n", out_dir, spl_name);
	fprintf (out, "l <- %d\n", mask.rows>mask.cols ? mask.rows : mask.cols);
	fprintf (out, "w <- 1.5 * l\n");
	fprintf (out, "h <- %.3f * w\n", hw_ratio);
	fprintf (out, "\n");
	fprintf (out, "ct <- read.table('%s/%s.scRNA2spatial.cell_contour.txt', header=F, sep='\\t')\n", out_dir, spl_name);
	fprintf (out, "expr <- read.table('%s/%s.scRNA2spatial.cell_contour.idx.txt', header=F, sep='\\t')\n", out_dir, spl_name);
	fprintf (out, "ncell <- nrow (expr)\n");
	fprintf (out, "ncluster <- ncol (expr) - 2\n");
	fprintf (out, "start <- expr$V1\n");
	fprintf (out, "end <- expr$V2\n");
	fprintf (out, "\n");

	for (i=0; i<ncls; ++i) {
		fprintf (out, "### Plot Cluster %d\n", i);
		fprintf (out, "val <- expr$V%d\n", i+3);
		fprintf (out, "max_val <- max (val)\n");
		fprintf (out, "if (max_val < %.6lf) {\n", min_maxval);
		fprintf (out, "  msg <- sprintf ('max score for cluster %d is %%.3f, which is too low. Skip ...', max_val)\n", i);
		fprintf (out, "  print (msg)\n");
		fprintf (out, "} else {\n");
		fprintf (out, "  plot (c(1,w), c(1,h), cex=0, xlab='', ylab='', main='', axes=F)\n");
		fprintf (out, "  polygon(c(0,0,w,w),c(0,h,h,0), col='%s', border=NA)\n", bg);
		fprintf (out, "  text (0.5*w, 0.9*h, 'Cluster %d', adj=0.5, col='%s')\n", i, text_color);
		fprintf (out, "\n");
		fprintf (out, "  for (i in 1:ncell) {\n");
		fprintf (out, "    y <- ct$V1[start[i]:end[i]]\n");
		fprintf (out, "    x <- ct$V2[start[i]:end[i]]\n");
		fprintf (out, "    y <- h - y\n");
		fprintf (out, "    x <- x + 0.05*w\n");
		fprintf (out, "    y <- y - 0.2*h\n");
		fprintf (out, "    col_idx <- as.integer(val[i]*1000/max_val) + 1\n");
		fprintf (out, "    if (col_idx < 1) {col_idx=1}\n");
		fprintf (out, "    if (col_idx > 1000) {col_idx=1000}\n");
		fprintf (out, "    polygon (x, y, col=pal[col_idx], border=NA)\n");
		fprintf (out, "  }\n");
		fprintf (out, "\n");
		fprintf (out, "  ly <- seq (h/3, 2*h/3, length.out=1001)\n");
		fprintf (out, "  for (i in 1:1000) {\n");
		fprintf (out, "    lxb <- 0.8*w\n");
		fprintf (out, "    lxe <- 0.8*w + 100\n");
    fprintf (out, "    rect (xleft=lxb, xright=lxe, ybottom=ly[i], ytop=ly[i+1], border=NA, col=pal[i])\n");
    fprintf (out, "  }\n");
    fprintf (out, "  text (0.8*w+150, ly[1], \"0\", adj=0, col='%s')\n", text_color);
    fprintf (out, "  ls <- sprintf (\"%%.1f\", max_val)\n");
    fprintf (out, "  text (0.8*w+150, ly[1000], ls, adj=0, col='%s')\n", text_color);
		fprintf (out, "}\n");
		fprintf (out, "\n");
	}
	fprintf (out, "dev.off()\n");
	fclose (out);

	sprintf (line, "Rscript %s/%s.scRNA2spatial.vec_img.plot.R", out_dir, spl_name);
	system (line);

	fprintf (stderr, "Plot %ld cells, cost %lds\n", n_plot, time(NULL)-time_beg);

	return 0;
}

// cell bin gem

typedef struct {
	int row, col;
	int gid;
	int exp;
	int cid;
	int blk;
} item_t;
MP_DEF (item, item_t);

static int
cmp (const void * a, const void * b)
{
	item_t * pa = (item_t *) a;
	item_t * pb = (item_t *) b;

	if (pa->row < pb->row)
		return -1;
	if (pa->row > pb->row)
		return 1;

	if (pa->col < pb->col)
		return -1;
	if (pa->col > pb->col)
		return 1;

	if (pa->cid < pb->cid)
		return -1;
	if (pa->cid > pb->cid)
		return 1;

	if (pa->gid < pb->gid)
		return -1;
	else
		return 1;
}

static int
extr_cellbin_gem_main (int argc, char * argv[])
{
	if (argc < 4) {
		fprintf (stderr, "\n");
		fprintf (stderr, "Usage:   cellbin_vec_plot gem [options] <in.gem> <mask.tif> <out.gem>\n");
		fprintf (stderr, "Options: -f  NUL  Fix gem x/y coordinates or not\n");
		fprintf (stderr, "\n");
		return 1;
	}

	char line[4096];
	char gene[4096];
	char out_gem[4096];
	int fix_xy;
	int copt;
	int i, j, k;
	int rf, cf;
	int row;
	int col;
	int n_labels;
	int n_genes;
	int n_cells;
	int val;
	int label;
	int col_min, col_max;
	int row_min, row_max;
	int * expr;
	int * cell_row;
	int * cell_col;
	int64_t new_cnt;
	double x, y;
	gzFile out;
	Mat mask;
	Mat labels;
	Mat stats;
	Mat centroids;
	str_t s;
	str_t * sptr;
	str_set_t * gene_names;
	xh_item_t * xh_ptr;
	str_hash_t * gene_hash;
	gzFile in;
	item_t * info;
	item_t * prev;
	item_t * dst;
	mp_t(item) * infos;

	fix_xy = 0;
	while ((copt = getopt(argc,argv,"f")) != -1) {
		if (copt == 'f')
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

	if (optind >= argc)
		err_mesg ("fail to get output gem file!");
	else
		strcpy (out_gem, argv[optind++]);

	if (str_cmp_tail(out_gem,".gz",3) != 0) {
		sprintf (line, "%s.gz", out_gem);
		out = ckgzopen (line, "w");
	} else
		out = ckgzopen (out_gem, "w");

	xcv_imread_gray (mask, mask_img);
	for (i=0; i<mask.rows; ++i)
		for (j=0; j<mask.cols; ++j)
			if (mask.at<uchar>(i,j) > 0)			
				mask.at<uchar>(i,j) = 255;
	n_labels = connectedComponentsWithStats (mask, labels, stats, centroids, 4, CV_32S);
	printf ("%d\n", n_labels);

	cell_row = (int *) ckalloc (n_labels, sizeof(int));
	cell_col = (int *) ckalloc (n_labels, sizeof(int));
	for (i=1; i<n_labels; ++i) {
		x = centroids.at<double>(i, 0);
		y = centroids.at<double>(i, 1);

		rf = (int) floor (y);
		cf = (int) floor (x);

		if (labels.at<int>(rf,cf) == i) {
			cell_row[i] = rf;
			cell_col[i] = cf;
		} else if (labels.at<int>(rf,cf+1) == i) {
			cell_row[i] = rf;
			cell_col[i] = cf + 1;
		} else if (labels.at<int>(rf+1,cf) == i) {
			cell_row[i] = rf + 1;
			cell_col[i] = cf;
		} else if (labels.at<int>(rf+1,cf+1) == i) {
			cell_row[i] = rf + 1;
			cell_col[i] = cf + 1;
		} else {
			row = stats.at<int>(i,1) + stats.at<int>(i,3)/2;
			for (j=stats.at<int>(i,0),k=0; k<stats.at<int>(i,2); ++j,++k)
				if (labels.at<int>(row,j) == i)
					break;
			if (labels.at<int>(row,j) == i) {
				cell_row[i] = row;
				cell_col[i] = j;
			} else {
				cell_row[i] = -1;
				cell_col[i] = -1;
			}
		}
	}

	n_cells = n_labels;

	if (fix_xy) {
		in = ckgzopen (gem_file, "r");
		row_min = col_min = INT_MAX;
		while (gzgets(in,line,4096))
			if (*line != '#')
				break;
		while (gzgets(in,line,4096)) {
			sscanf (line, "%*s %d %d", &col, &row);
			if (col < col_min) col_min = col;
			if (row < row_min) row_min = row;
		}
		gzclose (in);
	} else {
		row_min = col_min = 0;
	}

	in = ckgzopen (gem_file, "r");
	while (gzgets(in,line,4096))
		if (*line != '#')
			break;

	infos = mp_init (item, NULL);
	gene_hash = str_hash_init ();
	while (gzgets(in,line,4096)) {
		sscanf (line, "%s %d %d %d", gene, &col, &row, &val);
		row -= row_min;
		col -= col_min;
		label = labels.at<int>(row,col);
		if (label == 0) continue;
		assert (label < n_labels);

		if (cell_row[label]<0 || cell_col[label]<0)
			continue;

		s.s = gene;
		s.l = strlen (gene);
		xh_ptr = str_hash_add3 (gene_hash, &s);

		info = mp_alloc (item, infos);
		info->row = cell_row[label];
		info->col = cell_col[label];;
		info->gid = xh_ptr->id;
		info->cid = label;
		info->exp = val;
	}

	gzclose (in);

	mp_std_sort (item, infos, cmp);

	prev = mp_at (item, infos, 0);
	new_cnt = 0;
	val = prev->exp;
	for (i=1; i<infos->n; ++i) {
		info = mp_at (item, infos, i);
		if (info->row != prev->row
		 || info->col != prev->col
		 || info->gid != prev->gid
		 || info->cid != prev->cid) {
			dst = infos->pool + new_cnt++;
			dst->row = prev->row;
			dst->col = prev->col;
			dst->gid = prev->gid;
			dst->cid = prev->cid;
			dst->exp = val;

			val = 0;
			prev = info;
		}
		val += info->exp;
	}
	dst = infos->pool + new_cnt++;
	dst->row = prev->row;
	dst->col = prev->col;
	dst->gid = prev->gid;
	dst->cid = prev->cid;
	dst->exp = val;
	infos->n = new_cnt;

	gene_names = str_set_init ();
	xh_set_key_iter_init (xstr, gene_hash);
	while (sptr = xh_set_key_iter_next(xstr,gene_hash))
		str_set_add2 (gene_names, sptr->s, sptr->l);

	gzprintf (out, "#FileFormat=GEMv0.1\n");
	gzprintf (out, "#OffsetX=0\n#OffsetY=0\n");
	gzprintf (out, "geneID\tx\ty\tMIDCounts\tlabel\n");
	for (i=0; i<infos->n; ++i) {
		info = mp_at (item, infos, i);
		gzprintf (out, "%s\t%d\t%d\t%d\t%d\n",
				gene_names->pool[info->gid].s,
				info->col+1, info->row+1,
				info->exp, info->cid);
	}
	gzclose (out);

	return 0;
}

// Main

static int
main_usage (void)
{
	fprintf (stderr, "\n");
	fprintf (stderr, "Usage:   cellbin_vec_plot <Command> [Options]\n");
	fprintf (stderr, "Contact: chenxi1@genomics.cn\n");
	fprintf (stderr, "Version: 4.0\n");
	fprintf (stderr, "Command: cluster   Plot cluster on cellbin data\n");
	fprintf (stderr, "         gene      Display gene expression heatmap\n");
	fprintf (stderr, "         map       Visualize mapping scRNA to cellbin data\n");
	fprintf (stderr, "         gem       Extract cellbin Gem\n");
	fprintf (stderr, "         pre_ann   Create labeled gem from pre-annotated cell mask\n");
	fprintf (stderr, "         lasso     Extract in-lasso cells\n");
	fprintf (stderr, "\n");
	fprintf (stderr, "Updates: 3.4 -> 4.0\n");
	fprintf (stderr, "         1. add cmd <pre_ann> and <lasso>\n");
	fprintf (stderr, "\n");

	return 1;
}

static int
contour_main (int argc, char * argv[])
{
	if (argc < 4) {
		fprintf (stderr, "cellbin_vec_plot ctr <mask.tif> <cell.loc> <out.prefix>\n");
		return 1;
	}

	char line[4096];
	int i, j;
	int x, y;
	int val;
	int ncells;
	int64_t n_pnts;
	int64_t n_ctr_pnts;
	FILE * in;
	FILE * out;
	FILE * idx_out;
	Mat mask;
	Mat drawed;
	arr_t(i32) * ctr = arr_init (i32);
	arr_t(i32) * ctc = arr_init (i32);

	sprintf (line, "%s.cell_contour.txt", argv[3]);
	out = ckopen (line, "w");
	sprintf (line, "%s.cell_contour.idx.txt", argv[3]);
	idx_out = ckopen (line, "w");

	xcv_imread_gray (mask, argv[1]);
	for (i=0; i<mask.rows; ++i)
		for (j=0; j<mask.cols; ++j)
			if (mask.at<uchar>(i,j) > 0)
				mask.at<uchar>(i,j) = 255;
	drawed = Mat::zeros (mask.size(), CV_8UC1);

	ncells = 0;
	n_pnts = 0;
	in = ckopen (argv[2], "r");
	while (fgets(line,4096,in)) {
		sscanf (line, "%d %d", &x, &y);
		val = mask.at<uchar>(y,x);
		if (val==0 || cell_contour(mask,drawed,y,x,ctr,ctc,out)!=0) {
			fprintf (idx_out, "%ld\t%ld\n", n_pnts+1, n_pnts);
			continue;
		}

		fprintf (idx_out, "%ld\t%ld\n", n_pnts+1, n_pnts+ctr->n);
		n_pnts += ctr->n;

		if (++ncells % 1000 == 0)
			fprintf (stderr, "%d cells have been processed!\n", ncells);
	}
	printf ("%d cells\n", ncells);

	fclose (in);
	fclose (out);
	fclose (idx_out);

	return 0;
}

int
main (int argc, char * argv[])
{
	if (argc < 2)
		return main_usage ();

	int ret;

	l_bar = 500;
	data_type = 1;
	ex_xy = 0;
	run_norm = 0;
	max_expr = -1.0;
	hw_ratio = 1.0;
	bg = ALLOC_LINE;
	spl_name = ALLOC_LINE;
	out_dir = ALLOC_LINE;
	expr_file = ALLOC_LINE;
	rds_file = ALLOC_LINE;
	gem_file = ALLOC_LINE;
	gene_list_file = ALLOC_LINE;
	color_file = ALLOC_LINE;
	cluster_file = ALLOC_LINE;
	text_color = ALLOC_LINE;
	all_cell_pfile = ALLOC_LINE;
	mask_img = ALLOC_LINE;

	color_list = str_set_init ();

/*
 *   0
 * 3 X 1
 *   2
 */

	idx2ros[0] = -1;
	idx2ros[1] = 0;
	idx2ros[2] = 1;
	idx2ros[3] = 0;

	idx2cos[0] = 0;
	idx2cos[1] = 1;
	idx2cos[2] = 0;
	idx2cos[3] = -1;

	next_start[0] = 3;
	next_start[1] = 0;
	next_start[2] = 1;
	next_start[3] = 2;

	if (strcmp(argv[1], "cluster") == 0)
		ret = cluster_main (argc-1, argv+1);
	else if (strcmp(argv[1], "gene") == 0)
		ret = gene_main (argc-1, argv+1);
	else if (strcmp(argv[1], "map") == 0)
		ret = map_main (argc-1, argv+1);
	else if (strcmp(argv[1], "gem") == 0)
		ret = extr_cellbin_gem_main (argc-1, argv+1);
	else if (strcmp(argv[1], "pre_ann") == 0)
		ret = pre_anno_gem_main (argc-1, argv+1);
	else if (strcmp(argv[1], "lasso") == 0)
		ret = lasso_main (argc-1, argv+1);
	else if (strcmp(argv[1], "ctr") == 0)
		ret = contour_main (argc-1, argv+1);
	else {
		fprintf (stderr, "Error: unrecognized command '%s'\n", argv[1]);
		return main_usage ();
	}

	return ret;
}
