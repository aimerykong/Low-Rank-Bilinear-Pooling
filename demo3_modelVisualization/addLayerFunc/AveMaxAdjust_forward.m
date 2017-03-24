function now = AveMaxAdjust_forward(layer, pre, now)
%
% resulting into CxN matrix for classification in the loss layer following
% this one.
%
%
% Shu Kong @ UCI
% Aug 17, 2016

now.x = pre.x/784*1000;
return
%% forward pass -- parameters
inX = pre.x;
alpha = layer.alpha;
[h, w, c, N] = size(inX); % H  W  C  N
inX = permute( inX, [3, 1, 2, 4] ); % size is h, w, c, n
inX = reshape(inX, [c, h*w, N]);
%% fast pass
minRows = min(inX,[],2);
regX = bsxfun(@minus, inX, minRows);
expX = exp(alpha*regX);
expX = bsxfun(@rdivide, expX, sum(expX,2));
inX = inX.*expX;
%% slow pass
%{
for i = 1:N
    curX = inX(:,:,i);
    minRows = min(curX,[],2);
    regX = bsxfun(@minus, curX, minRows);
    expX = exp(alpha*regX);
    bsxfun(@rdivide, expX, sum(expX,2));
    inX(:,:,i) = curX.*expX;    
end
%}
%% return
inX = reshape(inX, [c, h, w, N]) ;
now.x = permute( inX, [2, 3, 1, 4] );


%% test code - equivalence of slow and fast version
%{
inX = rand(11,8,4,3);
alpha = 0.3;
[h, w, c, N] = size(inX); % H  W  C  N
inXSlow = permute( inX, [3, 1, 2, 4] ); % size is h, w, c, n
inXSlow = reshape(inXSlow, [c, h*w, N]);
for i = 1:N
    curX = inXSlow(:,:,i);
    minRows = min(curX,[],2);
    regX = bsxfun(@minus, curX, minRows);
    expX = exp(alpha*regX);
    bsxfun(@rdivide, expX, sum(expX,2));
    inXSlow(:,:,i) = curX.*expX;    
end
%
inXFast = permute( inX, [3, 1, 2, 4] ); % size is h, w, c, n
inXFast = reshape(inXFast, [c, h*w, N]);
minRows = min(inXFast,[],2);
regX = bsxfun(@minus, inXFast, minRows);
expX = exp(alpha*regX);
bsxfun(@rdivide, expX, sum(expX,2));
inXFast = inXFast.*expX;

%}
%% test code - correctness of calculation
%{
inX = rand(11,8,4,3);
alpha = 0.;
[h, w, c, N] = size(inX); % H  W  C  N
inXFast = permute( inX, [3, 1, 2, 4] ); % size is h, w, c, n
inXFast = reshape(inXFast, [c, h*w, N]);
minRows = min(inXFast,[],2);
regX = bsxfun(@minus, inXFast, minRows);
expX = exp(alpha*regX);
expX = bsxfun(@rdivide, expX, sum(expX,2));
inXFast = inXFast.*expX;
inXFast = reshape(inXFast, [c, h, w, N]);
inXFast = permute( inXFast, [2, 3, 1, 4] );
fprintf('#mismatch:%d\n', sum(inX(:)~=inXFast(:)));
fprintf('norm:%.8f\n', norm(inX(:)-inXFast(:)));
%}
%% blank ...
