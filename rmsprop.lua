--[[ An implementation of RMSprop

This is based on the original optim.rmsprop but modified to include centered version. 


ARGS:
- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- 'config.alpha'             : smoothing constant
- 'config.epsilon'           : value with which to initialise m
- 'config.weightDecay'       : weight decay
- 'cofnig.center'            : flags using centered rmsprop method.
- 'state'                    : a table describing the state of the optimizer;
                               after each call the state is modified
- 'state.m'                  : leaky sum of squares of parameter gradients,
- 'state.tmp'                : and the square root (with epsilon smoothing)
- 'state.m2'                 : leaky sum of parameter gradients. This is used to estmate variance.
RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update
]]

function rmsprop(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-2
   local alpha = config.alpha or 0.99
   local epsilon = config.epsilon or 1e-8
   local wd = config.weightDecay or 0
   local mfill = config.initialMean or 0
   local center = config.center or true
   -- (1) evaluate f(x) and df/dx
   local fx, dfdx = opfunc(x)

   -- (2) weight decay
   if wd ~= 0 then
      dfdx:add(wd, x)
   end

   -- (3) initialize mean square values and square gradient storage
   if not state.m then
     state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):fill(mfill)
      state.m2 = torch.Tensor():typeAs(x):resizeAs(dfdx):fill(mfill)
      state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
   end

   -- (4) calculate new (leaky) mean squared values
   state.m:mul(alpha)
   state.m:addcmul(1.0-alpha, dfdx, dfdx)
   -- If using centered method we need keep sum of gradients.
   if center then
     state.m2:mul(alpha)
     state.m2:add((1.0-alpha)*dfdx)
   end
   -- (5) perform update
   if center then
    -- rmsprop with variance of gradients (centered veresion)
    state.tmp:sqrt(state.m-torch.pow(state.m2,2)+epsilon)
   else
    -- rmsprop with magnitude of gradients
    state.tmp:sqrt(state.m):add(epsilon)
   end
   x:addcdiv(-lr, dfdx, state.tmp)

   -- return x*, f(x) before optimization
   return x, {fx}
end
