helper: helper.c
	gcc-11 -fPIC -shared -o helper.so helper.c -lm -fopenmp