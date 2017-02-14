# difference-target-propagation
A torch implementation of difference target propagation. You can find the original Theano implementation <a href='https://github.com/donghyunlee/dtp'>here</a>.


## Dependencies

nn
dataload
optim
dpnn
sys
nninit

## How to run on cpu

th diff_traget_prop_v3.lua


## How to run on gpu

th diff_traget_prop_v3.lua -gpu 1

## generate plots
gnuplot gnu_script.gnu
