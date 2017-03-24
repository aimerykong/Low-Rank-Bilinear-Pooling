function now = bisvm_posneg_UUreg_forward(layer, pre, now)
%
% forward pass of bilinear SVM layer
% compute
%   norm(Up_c'*X,'fro')^2 - norm(Un_c'*X,'fro')^2 + b, for all c = 1,...,C
% resulting into CxN matrix for classification in the loss layer following
% this one.
%
%
% Shu Kong @ UCI
% Sep, 2016

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
xin = xin.^2;

xin = reshape(xin, [rC, h*w, N]);
xin = squeeze( sum(xin, 2) );   % rC x 1 x N
xin = reshape(xin, [r, C, N]);  % r x C x N

xin = squeeze(sum(xin(1:r/2,:,:),1)) - squeeze(sum(xin(r/2+1:end,:,:),1));
%     xin = squeeze(sum(xin, 1)) ; % 1 x C x N

if numel(xin) > C
    xin = bsxfun(@plus, xin, b(:) ); % no bug?
else
    xin = bsxfun(@plus, xin(:), b(:) );
end
now.x = xin;



