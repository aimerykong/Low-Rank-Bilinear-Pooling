function net=addGlobalL2norm(net)
%% 
% The code is to globally normalize the data to have unit Frobenius norm.
% If the data is a vector, then the result is a unit length vector, in
% which case the normalization is identical to L2 normalization
%
% Copyright (C) 2016 Shu Kong.
% All rights reserved.
%
% This file is made available under the terms of the BSD license.

net.layers{end+1}=struct('type', 'custom',...
    'forward',  @GlobalL2norm_forward, ...
    'backward', @GlobalL2norm_backward, ...
    'name', 'GlobalL2');
