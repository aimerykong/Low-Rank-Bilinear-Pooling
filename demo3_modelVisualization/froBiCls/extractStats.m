% -------------------------------------------------------------------------
function stats = extractStats(net, opts, errors)
% -------------------------------------------------------------------------
stats.objective = errors(1) ;
for i = 1:numel(opts.errorLabels)
    stats.(opts.errorLabels{i}) = errors(i+1) ;
end

