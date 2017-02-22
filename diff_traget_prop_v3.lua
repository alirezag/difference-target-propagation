--[[
I give an implementation of difference target propagation.
The motivation is approximating backpropagation without depending
on differentiability of activation function or propagating the global error.



Author: Alireza Goudarzi
Email: alireza.goudarzi@riken.jp
copyright 2017, allrights reserved.


log -


22feb2017- 
adding orthogonalization similar to theano code. 

22feb2017- 
adding rmsprop with variance normalization. 

7feb2017-

the program here is completely flawed. I have to start from the scratch and see how to make it right. continue in Dev/7feb2017

31jan2017-

now using dtp_training_v3.lua which reorganizes the training epoch to math the original dtp so that first
inverse model is training on the entire data then the forward model and then iterate for 100 epochs. 

28jan2017-

added optim package for forward and inv model training. 

27jan2017-

the orthogonal initialization turned out to have a big impact (WHY)?
noise for inverse model and gpu support also added. 

To do: 0) done, add gpu support
       1) Optimization, several redundant model wide forwards.
       2) done, adding noise in training the inverse model as with the paper
       3) adding optim package so we can do adam or rmsprop training
       4) done, add orthogonal matrix initialization

23jan2017-

Now using approximate correction term 2h - g(h_i) as in the paper

23jan2017-

Adding Learning rate and tunable epochs for f and g model training.
The order of training matches the order of training on the paper.
right now with two hidden layer 240 each batch size 100, epoch size 5000
fLr=gLr=0.5 and fEpochs=gEpochs=50, cRate=0.01, 5000 training samples I get to
training accuracy of 0.85. optimization is normal SGD.



21jan2017-

refactoring and generalizing to L layer


20jan2017-

basic algorithm for 3 layers, no noise in inverse model, includes suggested correction. minimization works.
currently the order of training slightly deviates from the suggested work in the paper.

]]

require "nn"
dl = require "dataload"
require "optim"
require "dpnn" -- needed for nn.Convert
require "sys"
dofile('rmsprop.lua');
dofile('init_matrix.lua');
nninit = require 'nninit'
-- get options
cmd = torch.CmdLine();
cmd:text('Train simple network GPU benchmarking...');
cmd:text('Options');
cmd:option('-gpu',0,'Gpu device to use');
params = cmd:parse(arg)

print(params)


trainset, testset = dl.loadMNIST();


--define global parameters
maxEpoch = 100;
epochsize = 50000; batchsize = 100;
invNoiseSD=0.359829566008; --from the dtp file

--define main network parameters
 inputsize = 28*28; outputsize = 10;
 hiddensize = 240;
 L=7
 Lsize = {inputsize}
for i=2,L+1 do
Lsize[i] = hiddensize;
end
 fLR = 0.0148893490317;

dofile('define_forward_model.lua');

--define the inverse network parameters
gLR = 0.00501149118237;
cRate = 0.327736332653;

dofile('define_inverse_model.lua');

-- MSE criterion, used by forward f and inverse g models
MSECF = nn.MSECriterion(false)



if params.gpu>0 then
	print('Enabling GPU, running on device:', gpu);
	require "cutorch"
	require "cunn"
	cutorch.setDevice(params.gpu);

	-- take data to gpu
	trainset.inputs:cuda();
	trainset.targets:cuda();
	testset.inputs:cuda();
	testset.targets:cuda();

	-- convert model to gpu
	 for i,v in pairs(allFnets) do
		allFnets[i]:cuda();
	 end
	fcriterion:cuda();
	for i,v in pairs(allGnets) do 
		allGnets[i]:cuda();
    end

	-- get new model params
	for i=2,L+2 do
	  fparams, fparams_g = allFnets[i]:getParameters();
	  fgradi = allFnets[i].gradInput;
	  allFparams[i-1] = fparams; allFgrads[i-1] = fparams_g; allFgradinp[i-1] = fgradi;
	  allFoutputs[i-1] = 0;
	end
	for i=1,L-1 do
	  gparams,gparams_g = allGnets[i]:getParameters(); gradi = allGnets[i].gradInput;
	  allGparams[i] = gparams;
	  allGgrads[i] = gparams_g;
	  allGgradinp[i] = gradi ;
	  allGoutputs[i] = 0;
	end

	MSECF:cuda();
end


-- do training
dofile('dtp_training_v4.lua');


