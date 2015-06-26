.PHONY: all clean test re

all: knn

%: %.c
	gcc -Wall -O99 $< -lm -o $@

clean:
	rm -f knn

install: knn
	install -m 0755  knn /usr/local/bin/knn

test_%: knn
	./knn "$(patsubst test_%,%,$@)"
	gnuplot -e 'IN="$(patsubst test_%,%,$@).in"' plot_$(patsubst test_%,%,$@)
	gnuplot -e 'IN="$(patsubst test_%,%,$@).test"' plot_$(patsubst test_%,%,$@)
	gnuplot -e 'IN="$(patsubst test_%,%,$@).predic"' plot_$(patsubst test_%,%,$@)

test: knn test_a test_b test_c

re: clean all
