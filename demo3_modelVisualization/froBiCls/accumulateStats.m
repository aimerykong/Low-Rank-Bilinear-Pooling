% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

for s = {'train', 'val'}
    s = char(s) ;
    total = 0 ;
    
    % initialize stats stucture with same fields and same order as
    % stats_{1}
    stats__ = stats_{1} ;
    names = fieldnames(stats__.(s))' ;
    values = zeros(1, numel(names)) ;
    fields = cat(1, names, num2cell(values)) ;
    stats.(s) = struct(fields{:}) ;
    
    for g = 1:numel(stats_)
        stats__ = stats_{g} ;
        num__ = stats__.(s).num ;
        total = total + num__ ;
        
        for f = setdiff(fieldnames(stats__.(s))', 'num')
            f = char(f) ;
            stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;
            
            if g == numel(stats_)
                stats.(s).(f) = stats.(s).(f) / total ;
            end
        end
    end
    stats.(s).num = total ;
end

