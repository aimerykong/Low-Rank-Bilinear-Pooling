function pre = GlobalL2norm_backward(layer, pre, now)


dzdy = now.dzdx;
X = pre.x;
lambda = 1./( sqrt(sum(sum(sum(X.^2,1),2),3)) + 1e-10);

dzdx = sum(sum(sum(X.*dzdy,1),2),3);
dzdx = (lambda.^3) .* dzdx;
dzdx = bsxfun(@times, X, dzdx);
dzdx = bsxfun(@times, lambda, dzdy) - dzdx;
pre.dzdx = dzdx;
