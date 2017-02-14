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


--############################################
--######## The  actual train modules #########
--############################################


-- calclating forward model
function calculateForwardModel(inputs) 
	allFoutputs[1] = allFnets[1]:forward(inputs)
    for i=1,L+1 do
        allFoutputs[i+1] = allFnets[i+1]:forward(allFoutputs[i])
    end
end 

-- train inverse models
function trainInverseModels()
	for i=2,L do
        trainGnet(i);
    end
end


-- training the inverse model
function trainGnet(i)
	    if params.gpu>0 then
          allFoutputs_noise[i-1] = allFoutputs[i] + torch.Tensor(allFoutputs[i]:size(1),allFoutputs[i]:size(2)):normal(0,invNoiseSD):cuda();
        else
          allFoutputs_noise[i-1] = allFoutputs[i] + torch.Tensor(allFoutputs[i]:size(1),allFoutputs[i]:size(2)):normal(0,invNoiseSD);
        end
          FallFoutputs_noise[i-1] = allFnets[i+1]:forward(allFoutputs_noise[i-1]):clone();
      function g2eval(params)
             allGgrads[i-1]:zero();
             g2outputs = allGnets[i-1]:forward(FallFoutputs_noise[i-1]);
             loss_g2 = MSECF:forward(g2outputs,allFoutputs_noise[i-1]);
             dloss_g2 = MSECF:backward(g2outputs,allFoutputs_noise[i-1]);
             allGnets[i-1]:backward(FallFoutputs_noise[i-1],dloss_g2)
          return loss_g2,allGgrads[i-1]
          end

    optim.rmsprop(g2eval,allGparams[i-1],{learningRate=gLR, alpha  = 0.95, epsilon=0.001})
end


-- calculate inverse models (activation propagation)
function calculateInverseModels(targets)
	fcriterion:forward(allFoutputs[L+2],targets);
        allFnets[L+2]:zeroGradParameters();
        allFest[L] = allFoutputs[L+1] - cRate*allFnets[L+2]:backward(allFoutputs[L+1],fcriterion:backward(allFoutputs[L+2],targets));
    for i=L-1,1,-1 do
      allFest[i] = allFoutputs[i+1] - allGnets[i]:forward(allFoutputs[i+2]) + allGnets[i]:forward(allFest[i+1]);
    end
end

-- train forward models
function trainForwardModels(targets)

	i=L+2;
        -- ########### for difference target propagation
        function f4eval(params)
           allFgrads[i-1]:zero();
           allLoss[i-1] = fcriterion:forward(allFoutputs[i],targets);
           dloss_f4 = fcriterion:backward(allFoutputs[i],targets);
           allFgradinp[i-1] = allFnets[i]:backward(allFoutputs[i-1],dloss_f4)
        return allLoss[i-1], allFgrads[i-1]
        end
    optim.rmsprop(f4eval,allFparams[i-1],{learningRate=fLR, epsilon  = 0.95, alpha=0.001})

    for i=2,L+1 do
    -- ########### for difference target propagation
        function f4eval(params)
           allFgrads[i-1]:zero();
           allLoss[i-1] = MSECF:forward(allFoutputs[i],allFest[i-1]);
           dloss_f3 = MSECF:backward(allFoutputs[i],allFest[i-1]);
           allFnets[i]:backward(allFoutputs[i-1],dloss_f3)
    return allLoss[i-1], allFgrads[i-1]
        end
        optim.rmsprop(f4eval,allFparams[i-1],{learningRate=fLR, epsilon  = 0.95, alpha=0.001})
    end

end


--  calculate stats
function calculateStats()
-- evaluate empirical risk and confusion matrix
 cm = optim.ConfusionMatrix(10)
-- global
cm:zero();
calculateForwardModel(trainset.inputs);
trainloss = fcriterion:forward(allFoutputs[L+2],trainset.targets)/trainset.targets:size(1);
cm:batchAdd(allFoutputs[L+2], trainset.targets)
cm:updateValids();
trainVal = cm.totalValid;
cm:zero();
calculateForwardModel(testset.inputs);
testloss = fcriterion:forward(allFoutputs[L+2],testset.targets)/testset.targets:size(1);
cm:batchAdd(allFoutputs[L+2], testset.targets)
cm:updateValids();
testVal = cm.totalValid;
return trainloss, trainVal, testloss, testVal
end
