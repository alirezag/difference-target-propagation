--define mse
function mse (x) return torch.mean(torch.sum(torch.pow(x,2),2),1) end

--define the main network f
allFnets = {}; allFparams = {}; allFgrads = {}; allFgradinp = {}; allFoutputs = {};

allFnets[1] = nn.Sequential()
allFnets[1]:add(nn.Convert())
allFnets[1]:add(nn.View(inputsize))

for i=2,L+1 do
allFnets[i] = nn.Sequential();
allFnets[i]:add(nn.Linear(Lsize[i-1], Lsize[i]));--:init('weight',nninit.orthogonal,{torch.sqrt(6/(Lsize[i-1]+Lsize[i]))}));
allFnets[i].modules[1].weight = rand_ortho(Lsize[i-1],Lsize[i],torch.sqrt(6/(Lsize[i-1]+Lsize[i])));
allFnets[i].modules[1].bias:zero()
allFnets[i]:add(nn.Tanh())
fparams, fparams_g = allFnets[i]:getParameters();
fgradi = allFnets[i].gradInput;
allFparams[i-1] = fparams; allFgrads[i-1] = fparams_g; allFgradinp[i-1] = fgradi;
allFoutputs[i-1] = 0;
end


allFnets[L+2] = nn.Sequential();
allFnets[L+2]:add(nn.Linear(Lsize[L+1], 10));--:init('weight',nninit.orthogonal,{torch.sqrt(6/(Lsize[L+1]+10))}));
allFnets[L+2].modules[1].weight=rand_ortho(Lsize[L+1],10,torch.sqrt(6.0/(Lsize[L+1]+10.0)));

allFnets[L+2].modules[1].bias:zero()
allFnets[L+2]:add(nn.LogSoftMax())

i = L+2;
fparams, fparams_g = allFnets[i]:getParameters();
fgradi = allFnets[i].gradInput;
allFparams[i-1] = fparams; allFgrads[i-1] = fparams_g; allFgradinp[i-1] = fgradi;
allFoutputs[i-1] = 0;
fcriterion = nn.ClassNLLCriterion(false,false)

allFoutputs_noise ={};
FallFoutputs_noise = {};
allFest = {};
