/* Naive Bayes con Gaussiana - Guido Martínez */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h>
#include <errno.h>
#include <time.h>
#include <stdbool.h>

#define warn(s, ...)		\
	fprintf(stderr, "%s.%i: " s, __func__, __LINE__ , ##__VA_ARGS__); \

#define fail(s, ...)		\
({				\
 	warn(s, ##__VA_ARGS__);	\
	exit(1);		\
})

struct cfg {
	int inputs;	/* Number of input variables, a.k.a dimensionality */
	int classes;	/* Number of classes */
	int patterns;	/* How many examples are in the file */
	int tests;	/* How many patterns are in the test file */

	int seed;	/* Random seed, used for shuffling the examples */
	int cv_split;	/* Percentage of training data used for validation */

	int minK;
	int maxK;

	bool ball;
	double minD, maxD, step;

	/* Not configurable, calculated in base of others */
	int train_patterns;	/* How many patterns are used for training
				   (= patterns * cv_split / 100) */
	int valid_patterns;	/* How many patterns are used for validation
				   (= patterns - train_patterns) */
} cfg = {
	.cv_split = 20,
	.minK = 3,
	.maxK = 7,
};

static const struct option long_options[] = {
	{"inputs",	required_argument, 0, 'i'},
	{"classes",	required_argument, 0, 'c'},
	{"patterns",	required_argument, 0, 'p'},
	{"tests",	required_argument, 0, 't'},
	{"seed",	required_argument, 0, 's'},
	{"split",	required_argument, 0, 'v'},
	{"k",		required_argument, 0, 'k'},
	{"ball",	no_argument,       0, 'b'},
	{"d",		required_argument, 0, 'd'},
};

static void set_opt(char c, char *arg)
{
	switch (c) {
	case 'i':
		cfg.inputs = atoi(arg);
		break;
	case 'c':
		cfg.classes = atoi(arg);
		break;
	case 'p':
		cfg.patterns = atoi(arg);
		break;
	case 't':
		cfg.tests = atoi(arg);
		break;
	case 's':
		cfg.seed = atoi(arg);
		srand(cfg.seed);
		break;
	case 'v':
		cfg.cv_split = atoi(arg);
		break;
	case 'k':
		if (2 != sscanf(arg, "%i..%i", &cfg.minK, &cfg.maxK))
			fail("Invalid interval for K\n");
		break;
	case 'b':
		cfg.ball = true;
		break;
	case 'd':
		if (3 != sscanf(arg, "%lf..%lf..%lf", &cfg.minD, &cfg.step,
						      &cfg.maxD))
			fail("Invalid interval for D\n");
		break;
	case '?':
		fail("usage....\n");
	}
}

static void parse_opts(int argc, char **argv)
{
	int c, idx;

	optind = 1;

	while (c = getopt_long(argc, argv, "", long_options, &idx), c != -1)
		set_opt(c, optarg);
}

static void sort_opts(int argc, char **argv)
{
	int c, idx;
	while (c = getopt_long(argc, argv, "", long_options, &idx), c != -1) {
		if (c == '?')
			fail("usage....\n");
	}
}

static char lookup_cfg(char *key)
{
	const struct option *p;

	for (p = long_options; p->name; p++)
		if (!strcmp(p->name, key))
			return p->val;

	return 0;
}

static void read_cfg(char *stem)
{
	char fname[80];
	char key[200];
	char val[200];

	sprintf(fname, "%s.cfg", stem);
	FILE *cfg = fopen(fname, "r");

	if (!cfg && errno == ENOENT)
		return;
	else if (!cfg)
		fail("Opening cfg file failed (%s, %i)\n", fname, errno);

	while (fscanf(cfg, "%[^=]=%s\n", key, val) == 2) {
		char c = lookup_cfg(key);

		if (!c)
			fail("Unknown option: %s\n", key);

		set_opt(c, val);
	}

	fclose(cfg);
}

static void show_cfg()
{
	cfg.train_patterns = cfg.patterns * (100 - cfg.cv_split) / 100;
	cfg.valid_patterns = cfg.patterns - cfg.train_patterns;

	printf("Total patterns = %i\n", cfg.patterns);
	printf("Train patterns = %i\n", cfg.train_patterns);
	printf("Validation patterns = %i\n", cfg.valid_patterns);
}

double **train_data;

int read_csv(FILE *f, int rows, int cols, double **mat)
{
	int i, j;
	int ret;
	double t;

	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			if (j != cols - 1)
				ret = fscanf(f, "%lf,", &t);
			else
				ret = fscanf(f, "%lf\n", &t);

			if (ret != 1) {
				printf("fscanf returned %i\n", ret);
				return 1;
			}

			mat[i][j] = t;
		}
	}

	return 0;
}

double **alloc_matrix(int rows, int cols)
{
	double **ret;
	int i;

	ret = malloc(sizeof(double*) * rows);
	for (i = 0; i < rows; i++) {
		ret[i] = malloc(sizeof(double) * cols);
		memset(ret[i], 0, sizeof(double) * cols);
	}

	return ret;
}

int read_data_file(char *stem, char *suffix, double **(*mat), int rows)
{
	char fname[80];
	FILE *f;
	int ret;
	double **m;

	sprintf(fname, "%s.%s", stem, suffix);

	m = alloc_matrix(rows, cfg.inputs + 1);

	f = fopen(fname, "r");
	if (!f) {
		warn("Could not open %s\n", fname);
		return 1;
	}

	ret = read_csv(f, rows, cfg.inputs + 1, m);
	if (ret) {
		warn("read_csv failed (%i)\n", ret);
		return 1;
	}

	*mat = m;

	return 0;
}

int shuffle(double **m, int rows, int cols)
{
	int i, p;
	int j;
	double d;

	for (i = 1; i < rows; i++) {
		p = rand() % (i + 1);

		if (p == i)
			continue;

		/* Swap rows 'i' and 'p' */
		for (j = 0; j < cols; j++) {
			d = m[i][j];
			m[i][j] = m[p][j];
			m[p][j] = d;
		}
	}

	return 0;
}

double dist2(int len, double *v1, double *v2)
{
	double ret;
	int i;

	ret = 0;

	for (i = 0; i < len; i++)
		ret += (v1[i] - v2[i])*(v1[i] - v2[i]);

	return ret;
}

double dist(int len, double *v1, double *v2)
{
	return sqrt(dist2(len, v1, v2));
}

int predict_one(int K, double *vec, int len)
{
	int *nearest = alloca(sizeof(int) * (K + 1));
	double *dists = alloca(sizeof(double) * (K + 1));
	double *closest_dists = alloca(sizeof(double) * (cfg.classes));
	int i, j;
	int c_max = -1, c_count;
	int c, t;

	for (i = 0; i < K; i++)
		dists[i] = HUGE_VAL;

	for (i = 0; i < cfg.train_patterns; i++) {
		double d;

		d = dist2(len, vec, train_data[i]);

		/* Add to the bottom and sink them */
		dists[K] = d;
		nearest[K] = i;

		for (j = K; j > 0 && dists[j] < dists[j - 1]; j--) {
			double s;

			s = dists[j - 1];
			dists[j - 1] = dists[j];
			dists[j] = s;

			s = nearest[j - 1];
			nearest[j - 1] = nearest[j];
			nearest[j] = s;
		}
	}

	c_count = -1;
	for (c = 0; c < cfg.classes; c++) {
		closest_dists[c] = HUGE_VAL;

		t = 0;
		for (i = 0; i < K; i++) {
			if (train_data[nearest[i]][cfg.inputs] == c) {
				t++;

				if (dists[i] <= closest_dists[c])
					closest_dists[c] = dists[i];
			}
		}

		/*
		 * Si hay un empate, tomamos la clase que haya tenido
		 * su vecino mas cercano a menor distancia
		 */
		if (t > c_count ||
			(t == c_count
				 && closest_dists[c] < closest_dists[c_max])) {
			c_count = t;
			c_max = c;
		}
	}

	return c_max;
}

int predict_one_ball(double D, double *vec, int len)
{
	double *closest_dists = alloca(sizeof(double) * cfg.classes);
	int *count = alloca(sizeof(int) * cfg.classes);
	int winner;
	int c, i;

	for (i = 0; i < cfg.classes; i++) {
		count[i] = 0;
		closest_dists[i] = HUGE_VAL;
	}

	for (i = 0; i < cfg.train_patterns; i++) {
		double d;
		int c;

		d = dist2(len, vec, train_data[i]);
		c = train_data[i][cfg.inputs];

		if (d <= D * D)
			count[c]++;

		if (d < closest_dists[c])
			closest_dists[c] = d;
	}

	winner = 0;
	for (c = 1; c < cfg.classes; c++) {
		/*
		 * Buscar la clase con mas "votos", en caso de empate
		 * usar vecino mas cercano (incluso si no entraba en la bola)
		 */
		if (count[c] > count[winner] ||
				(count[c] == count[winner] &&
				 closest_dists[c] < closest_dists[winner]))
			winner = c;
	}

	return winner;
}

int do_predicts(double *_err, int K, int rows, double **m, char *fname)
{
	int i, c;
	double err = 0;
	FILE *f;
	int a;

	if (fname)
		f = fopen(fname, "w");
	else
		f = fopen("/dev/null", "w");

	for (i = 0; i < rows; i++) {
		c = predict_one(K, m[i], cfg.inputs);

		if (c != m[i][cfg.inputs])
			err++;

		for (a = 0; a < cfg.inputs; a++)
			fprintf(f, "%lf,", m[i][a]);
		fprintf(f, "%i\n", c);
	}

	err /= rows;

	*_err = err;

	return 0;
}

int do_predicts_ball(double *_err, double D, int rows, double **m, char *fname)
{
	int i, c;
	double err = 0;
	FILE *f;
	int a;

	if (fname)
		f = fopen(fname, "w");
	else
		f = fopen("/dev/null", "w");

	for (i = 0; i < rows; i++) {
		c = predict_one_ball(D, m[i], cfg.inputs);

		if (c != m[i][cfg.inputs])
			err++;

		for (a = 0; a < cfg.inputs; a++)
			fprintf(f, "%lf,", m[i][a]);
		fprintf(f, "%i\n", c);
	}

	err /= rows;

	*_err = err;

	return 0;
}

static void start_rand()
{
	struct timespec tp;

	clock_gettime(CLOCK_REALTIME, &tp);

	srand(tp.tv_sec * 1000 + tp.tv_nsec / 1000000);
}

int do_knn(double **d, double **t, char *stem)
{
	double err, minErr = HUGE_VAL;
	char predic_file[80];
	int bestK = -1;
	int k;

	if (cfg.minK != cfg.maxK) {
		if (cfg.maxK > cfg.train_patterns) {
			warn("maxK is too big, truncating to # of train patterns (%i)\n",
					cfg.train_patterns);
			cfg.maxK = cfg.train_patterns;
		}

		if (cfg.cv_split == 0)
			fail("Cannot optimize K with validation split = 0\n");

		for (k = cfg.minK; k <= cfg.maxK; k++) {
			do_predicts(&err, k, cfg.valid_patterns,
					d + cfg.train_patterns, NULL);

			if (err < minErr) {
				minErr = err;
				bestK = k;
			}
		}

		printf("best K = %i\n", bestK);
	} else {
		bestK = cfg.minK;
	}

	do_predicts(&err, bestK, cfg.train_patterns, d, NULL);
	printf("Error sobre TRAIN:	%lf\n", err);

	if (cfg.cv_split > 0) {
		do_predicts(&err, bestK, cfg.valid_patterns, d + cfg.train_patterns, NULL);
		printf("Error sobre VALID:	%lf\n", err);
	}

	sprintf(predic_file, "%s.predic", stem);

	do_predicts(&err, bestK, cfg.tests, t, predic_file);
	printf("Error sobre TEST:	%lf\n", err);

	return 0;
}

int do_ball(double **d, double **t, char *stem)
{
	double err, minErr = HUGE_VAL;
	char predic_file[80];
	double bestD = 0;
	int k;
	double D;

	for (D = cfg.minD; D <= cfg.maxD; D += cfg.step) {
		if (cfg.cv_split == 0)
			fail("Cannot optimize D with validation split = 0\n");

		for (k = cfg.minK; k <= cfg.maxK; k++) {
			do_predicts_ball(&err, D, cfg.valid_patterns,
					d + cfg.train_patterns, NULL);

			if (err < minErr) {
				minErr = err;
				bestD = D;
			}
		}

	}
	printf("best D = %f\n", bestD);

	do_predicts_ball(&err, bestD, cfg.train_patterns, d, NULL);
	printf("Error sobre TRAIN:	%lf\n", err);

	if (cfg.cv_split > 0) {
		do_predicts_ball(&err, bestD, cfg.valid_patterns, d + cfg.train_patterns, NULL);
		printf("Error sobre VALID:	%lf\n", err);
	}

	sprintf(predic_file, "%s.predic", stem);

	do_predicts_ball(&err, bestD, cfg.tests, t, predic_file);
	printf("Error sobre TEST:	%lf\n", err);

	return 0;
}

int main(int argc, char **argv)
{
	double **d;
	double **t;
	char stem[80];
	int ret;

	start_rand();

	if (argc < 2)
		fail("uso: %s <archivo>\n", argv[0]);

	/* UGH */
	sort_opts(argc, argv);

	/* Borrar .in al final si existe */
	strcpy(stem, argv[optind]);
	if (!strcmp(stem + strlen(stem) - 3, ".in"))
		stem[strlen(stem) - 3] = 0;

	read_cfg(stem);

	parse_opts(argc, argv);

	show_cfg();

	ret = read_data_file(stem, "in", &d, cfg.patterns);
	if (ret)
		fail("read_data failed (%i)\n", ret);

	ret = shuffle(d, cfg.patterns, cfg.inputs + 1);
	if (ret)
		fail("shuffle failed (%i)\n", ret);

	train_data = d;

	ret = read_data_file(stem, "test", &t, cfg.tests);
	if (ret)
		fail("read_test failed (%i)\n", ret);

	if (cfg.ball)
		return do_ball(d, t, stem);
	else
		return do_knn(d, t, stem);

	return 0;
}
