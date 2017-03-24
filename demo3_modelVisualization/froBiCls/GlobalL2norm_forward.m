function now = GlobalL2norm_forward(layer, pre, now)


% X = pre.x;
% szX = [size(X,1), size(X,2), size(X,3), size(X,4)];
% X = reshape(X, [ prod(szX(1:3)) szX(4)]);
% X = bsxfun(@rdivide, X, sqrt(sum(X.^2, 1)));
% X = reshape(X, szX);
% now.x = X;

now.x = bsxfun(@rdivide, pre.x, sqrt( sum(sum(sum(pre.x.^2,1),2),3) ) );


