function now = addFroNormAct_forward(layer, pre, now)
%
%
% Shu Kong @ UCI
% Oct, 2016

%%
%     inX = pre.x{1};
inX = pre.x;
% if GPU is used
gpuMode = isa(inX, 'gpuArray');

W = layer.weights{1}; % reducedDim(512) x r*C (20dim*200classes), first/second half->pos/neg part
b = layer.weights{2}; % 1 x C
r = layer.rDim;
C = layer.nClass;

[~, rC] = size(W);
assert(rC == r*C, 'incorrect weights at the biCls layer!');

[h, w, c, N] = size(inX); % H  W  C  N

xin = permute( inX, [3, 1, 2, 4] ); % size is h, w, c, n
xin = reshape(xin, [c, h*w*N]);

xin = W'*xin; % rC x h*w*N
% xin = reshape(xin, [rC, h*w, N]);
% xin = permute(xin, [2, 1, 3] ); % size is [h*w, rC, N]
% xin = reshape(xin, [h, w, rDim, nclass, N]);

now.x = xin;



