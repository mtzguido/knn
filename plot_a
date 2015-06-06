set size square
set datafile separator ","
set terminal png size 900,900 enhanced font "Helvetica,13"
set output (IN . '.png')

set xrange [-8:8]
set yrange [-8:8]

plot IN using 1:( $3 == 0 ? $2 : 1/0 ) title 'class 0', \
     IN using 1:( $3 > 0 ? $2 : 1/0 ) title 'class 1'


