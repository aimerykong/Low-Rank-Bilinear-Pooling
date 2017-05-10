function now = scaledSqrt_forward(layer, pre, now)
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
in = pre.x;
in = in*scaleFactor;
% in = sign(in) .* sqrt(abs(in)+ep);
in = sign(in) .* sqrt(abs(in));
now.x = in / scaleFactor;
%}
%% denominator as scaling factor
in = pre.x;
in = sign(in) .* sqrt(abs(in));
now.x = in / sqrt(scaleFactor);


