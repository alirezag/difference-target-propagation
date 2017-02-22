--define the inverse network g


-- we assume all hidden layers have the same dimensions
-- start from the top hidden layer (M-2) and work backward.
-- assume a reshape and view layer in the first two layers.
-- therefore the first layer hidden activation are at layer 4.

allGnets = {}; allGparams = {}; allGgrads = {}; allGgradinp = {}; allGoutputs = {};
Goptstate = {};

for i=1,L-1 do



allGnets[i] = nn.Sequential();
allGnets[i]:add(nn.Linear(hiddensize,hiddensize));--:init('weight',nninit.orthogonal,{torch.sqrt(6/(hiddensize+hiddensize))}));
allGnets[i].modules[1].weight = rand_ortho(hiddensize,hiddensize,torch.sqrt(6/(hiddensize+hiddensize)));

allGnets[i].modules[1].bias:zero();
allGnets[i]:add(nn.Tanh());
gparams,gparams_g = allGnets[i]:getParameters(); gradi = allGnets[i].gradInput;
allGparams[i] = gparams;
allGgrads[i] = gparams_g;
allGgradinp[i] = gradi ;
allGoutputs[i] = 0;
end
