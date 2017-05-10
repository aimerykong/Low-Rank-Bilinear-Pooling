function now = biCls_forward(layer, pre, now)    
%
% forward pass of bilinear SVM layer
% compute 
%   || U_c'*X ||_fro^2 + b, for all c = 1,...,C
% resulting into CxN matrix for classification in the loss layer following
% this one.
%
%
% Shu Kong @ UCI
% June 17, 2016

%%
    W = layer.weights{1}; % reducedDim(512) x r*C (20dim*200classes)
    b = layer.weights{2}; % 1 x C
    r = layer.rDim;
    C = layer.nClass;
    
    [~, rC] = size(W);
    assert(rC == r*C, 'incorrect weights at the biCls layer!');
    
    [h, w, c, N] = size(pre.x); % H  W  C  N
    
    xin = permute( pre.x, [3, 1, 2, 4] ); % size is h, w, c, n
    xin = reshape(xin, [c, h*w*N]); 
    
    xin = W'*xin; % rC x h*w*N
    xin = xin.^2;
    
    xin = reshape(xin, [rC, h*w, N]);
    xin = squeeze( sum(xin, 2) );   % rC x 1 x N
    xin = reshape(xin, [r, C, N]);  % r x C x N
    xin = squeeze(sum(xin, 1)) ; % 1 x C x N   
    xin = bsxfun(@plus, xin, b(:) );
    now.x = xin;              
end  


