--[[

log-


14feb2017:

making the code modular

13feb2017:

adding loggers


12feb2017:

more experiment with Theano DTP showes that even if I turned their costs to all sums it still converges rapidly for 
7 layer nets. this means their implementation of rmsprop should be slightly different than torch. That means 
adjusting the LR rate should hep in torch. I disabled all normalization in the training code and in cost function
and divided original theano learning rates by 3 and I now get similar learning curves to theano with 7 layer network.

11feb2017:

i realized that even for MSECF I have to turng off the normalization for the system to work. but I do have to normalize by
sample size in NLL. 

10feb2017:

I will not normalize NLL for the training but I hve to normalize the loss I report.

10feb2017: 

the basic code is working. the problem was that the criterion functions are averaged over dimensions which changes the meaninig.:w

now generalizaing to N layer.

7feb2017:

reimplementing the dtp in torch just for 3 layers and making sure it works as well.

31jan2017:

reorganizing and epoch so a single epoch includes full training of inverse model on the dataset then forward model.
this follows the original dtp protocol

28jan2017:

added the optim package so we can use different optimization algorithms. 

27jan2017: 

1) Reverted back from the approximate difference 2h-g(h_i) to the origianl differece formula
2) added noise for inverse model training

]]

require 'dtp_training_utils.lua'

logger = optim.Logger('results.log')
logger:setNames{'epoch', 'Train Loss','Train acc.','Test Loss','Test acc.','time'};
print('Epoch, train loss, train acc., test loss, test acc, time');

-- to logg results.
allLoss = {}; allLossMy = {};

for epochi = 1,maxEpoch do
local start =sys.tic();

for j,inputs, targets in trainset:sampleiter(batchsize,epochsize) do
	-- calculate forward model f
	calculateForwardModel(inputs);
	-- train inverse model
	trainInverseModels();
end

-- train forward model
for j,inputs, targets in trainset:sampleiter(batchsize,epochsize) do
    -- calculate forward model f
	calculateForwardModel(inputs);
	-- calculate inverse moderls g
    calculateInverseModels(targets)
	-- train forward models
	trainForwardModels(targets);	
end

trainloss, trainVal, testloss, testVal = calculateStats();
time = sys.toc();
logger:add{epochi,trainloss,trainVal,testloss,testVal,time}
print( string.format('%5d, %9.5f,   %.4f,   %9.5f,   %.4f,   %2.2f',epochi,trainloss,trainVal,testloss, testVal,  time));
end


