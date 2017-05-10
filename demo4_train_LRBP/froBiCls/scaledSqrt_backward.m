function pre = scaledSqrt_backward(layer, pre, now)
% set the ep LARGE enough to avoid gradient explosion
% set scaleFactor reasonably LARGE enough to deal with burstiness
%
% Shu Kong @ UCI
% Jun. 2016

%%
if isfield(layer, 'scaleFactor')
    scaleFactor = layer.scaleFactor;
else
    scaleFactor = 10000;
end

if isfield(layer, 'ep')
    ep = layer.ep;
else
    ep = 1e-1;
end

%% scaled sqrt root
%{
%pre.dzdx = now.dzdx .* 0.5 ./ (sqrt(abs(pre.x)) + ep); % the original
pre.dzdx = now.dzdx .* 0.5 ./ (sqrt(scaleFactor.*abs(pre.x)) + ep);
%}

%% denominator as scaling factor
pre.dzdx = now.dzdx .* (0.5/sqrt(scaleFactor)) ./ (sqrt(abs(pre.x)) + ep);

