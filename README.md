# Difference Target Propagation

I give a torch implementation of difference target propagation.
The motivation is approximating backpropagation without depending
on differentiability of activation function or propagating the global error.
<br>
Author: Alireza Goudarzi <br>
Email: alireza.goudarzi@riken.jp <br>
<br>

You can find the original Theano implementation <a href='https://github.com/donghyunlee/dtp'>here</a>.
You can find the arxiv paper <a href='https://arxiv.org/abs/1412.7525'>here</a>.


## Dependencies

The code is written for Torch7. The following packages are required: <br>
nn<br>
dataload<br>
optim<br>
dpnn<br>
sys<br>
nninit<br>

## How to 

run on CPU:

   th diff_traget_prop_v3.lua

run on GPU:

   th diff_traget_prop_v3.lua -gpu 1

generate plots:

   gnuplot gnu_script.gnu

## Notes

Loss normalization between Theano's original code and Torch packages are different and therefore the 
convergence of the model in Theano and Torch is different even when using identical metaparameters. 

I experimentally discovered that if I disable normalization in Torch and divide learning rates by 3 
I get similar convergence speed to Theano implementation. 

No momentum is used in the optimization. In each epoch, first the inverse model is trained on the entire
dataset and then the forward model is trained (as in the original implementation).


## Results

Using 50,000 MNIST examples to train and 10,000 to test the performance. 
<img src='result_loss.png'>
<br>
<img src='result_acc.png'>

## Comparing with original Theano implementation of DTP

To directly compare the results of Torch vs. Theano implementations you can turn momentum off on Theano implementation. You need to use the original learning rates in Torch but use no loss normalization for MSE criterion and manually do batchsize normalization for NLL criterion in Toch. Thea means you have to divide the result of the forward and backward calls to NLL by the batchsize. Here is a comparison between Theano and Torch implementations under these conditions showing similar convergence rates. The data is the test error rate on MNIST dataset. (50K examples to train, 10K to test). 

<img src='torch_theano_dtp_result_error2.png'>
