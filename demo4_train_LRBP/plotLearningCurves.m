function plotLearningCurves(stats)

if ~isempty(stats) && isfield(stats, 'val') && ~isempty(stats.val)
    plots = setdiff(...
        cat(2,...
        fieldnames(stats.train)', ...
        fieldnames(stats.val)'), {'num', 'time'}) ;
    for p = plots
        p = char(p) ;
        values = zeros(0, length(stats.train)) ;
        leg = {} ;
        for f = {'train', 'val'}
            f = char(f) ;
            if isfield(stats.(f), p)
                tmp = [stats.(f).(p)] ;
                values(end+1,:) = tmp(1,:)' ;
                leg{end+1} = f ;
            end
        end
        subplot(1,numel(plots),find(strcmp(p,plots))) ;
        plot(1:size(values,2), values(:, 1:end)','.-') ; % don't plot the first epoch
        xlabel('epoch') ;
        [~,minIdx] = min(values(2,:));
        [~,minIdxTr] = min(values(1,:));
        % title(sprintf('tsErr%.4f (%d) trErr%.4f', min(values(2,:)), minIdx, min(values(1,:)))) ;
        title(sprintf('%s ts%.4f (%d) tr%.4f (%d) ', p, min(values(2,:)), minIdx, min(values(1,:)),minIdxTr)) ;
        legend(leg{:},'location', 'SouthOutside') ;
        grid on ;
    end
    drawnow ;
end