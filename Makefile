.PHONY: all clean

all: knn

%: %.c
	gcc -g -Wall -O99 $< -lm -o $@

clean:
	rm -f knn

install: knn
	cp knn /usr/local/bin/
