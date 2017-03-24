function now = learnablePowerNormLayer4biCls_forward(layer, pre, now)
%
%
% Shu Kong @ UCI
% July 2016

%%
scaleFactor = layer.scaleFactor;
rootFactor = layer.weights{1};

% rootFactor = ones(512,1)*0.5;

in = pre.x;
[h, w, c, N] = size(pre.x); % H  W  C  N

rootFactor = reshape(rootFactor, [1, 1, c]);
rootFactor = repmat(rootFactor, [h, w, 1, N]);
now.x = sign(in).* (abs(in).^rootFactor);
now.x = now.x ./ (scaleFactor.^rootFactor);


% in = pre.x;
% in = sign(in) .* sqrt(abs(in));
% now.x = in / sqrt(scaleFactor);
