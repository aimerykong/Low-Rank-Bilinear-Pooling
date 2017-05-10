function pre = GlobalL2norm_backward(layer, pre, now)


%%{
% my code
dzdy = now.dzdx;
X = pre.x;
lambda = 1./( sqrt(sum(sum(sum(X.^2,1),2),3)) + 1e-10);

dzdx = sum(sum(sum(X.*dzdy,1),2),3);
dzdx = (lambda.^3) .* dzdx;
dzdx = bsxfun(@times, X, dzdx);
dzdx = bsxfun(@times, lambda, dzdy) - dzdx;
pre.dzdx = dzdx;
%}

%{
% Yang Gao's code
X = pre.x;
dzdy = now.dzdx;
[h,w,c,N] = size(X);
X = reshape(X,[h*w*c,N]);
dzdy = reshape(dzdy,[h*w*c, N]);

lambda=1./(sqrt(sum(X.^2, 1)) + 1e-10);

dzdx = bsxfun(@times, lambda, dzdy) - bsxfun(@times, X, (lambda.^3) .* sum(X.*dzdy, 1));
dzdx = reshape(dzdx, [h,w,c,N]);
pre.dzdx = dzdx; % reshape(dzdx, 1, 1, size(dzdx, 1), size(dzdx, 2));
%}




% dzdy = squeeze(dzdy);
% X = squeeze(pre.x);
% lambda = 1./(sqrt(sum(X.^2, 1)) + 1e-10);
% dzdx = bsxfun(@times, lambda, dzdy) - bsxfun(@times, X, (lambda.^3) .* sum(X.*dzdy, 1));
% pre.dzdx = reshape(dzdx, 1, 1, size(dzdx, 1), size(dzdx, 2));

