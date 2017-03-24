function pre = learnablePowerNormLayer4biCls_backward(layer, pre, now)
%
%
% Shu Kong @ UCI
% July 2016

%%
ep = layer.ep;
scaleFactor = layer.scaleFactor;

[h, w, c, N] = size(pre.x); % H  W  C  N

rootFactor = layer.weights{1};
rootFactor = reshape(rootFactor, [1, 1, c]);

%% dzdx
A = repmat(rootFactor, [h, w, 1, N]);
pre.dzdx = now.dzdx .* (A./ (scaleFactor.^A)) ./ ( abs(pre.x).^(1-A) + ep); % delta

% pre.dzdx = now.dzdx .* (0.5/sqrt(scaleFactor)) ./ (sqrt(abs(pre.x)) + ep);

%%{
%% dzdw -- derivative w.r.t exponential factor
A = bsxfun(@power, abs(pre.x), rootFactor);
B = abs(pre.x);
B(B==0) = 1;

A = gather(now.dzdx) .* log( B ) .* A;
A = mean(gather(A),4);
A = reshape(A, [size(A,1)*size(A,2), size(A,3)]);
A = mean(A, 1);
A = squeeze(A(:));
pre.dzdw = {A};
%}

%% visualize the gradient
% T = reshape(pre.dzdw, [512,512]);
% figure(1);
% imagesc(T), colorbar;

