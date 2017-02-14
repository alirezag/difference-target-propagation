#!/usr/bin/gnuplot -persist
set terminal png
set output 'result_loss.png'
set xlabel 'epochs'
set ylabel 'loss'
plot "results.log" using 1:2 title "train loss" with linespoints, "results.log" using 1:4 title "test loss" with linespoints
set output 'result_acc.png'
set xlabel 'epochs'
set ylabel 'accuracy'
plot "results.log" using 1:3 title "train acc." with linespoints, "results.log" using 1:5 title "test acc." with linespoints
