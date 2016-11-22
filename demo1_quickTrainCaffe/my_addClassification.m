function net = my_addClassification(net, lossType)
    % convert the 'SVM' and 'LR' to internal representation
    if strcmp(lossType, 'LR')
        lossType='softmaxlog';
    elseif strcmp(lossType, 'SVM')
%         lossType='mhinge';
        lossType='hinge';
    else
        error('unknown loss type');
    end

    
    % add the last softmax classification layer
    net.layers{end+1}=struct('type', 'loss',...
                             'name', 'final_loss', ...
                             'lossType', lossType);
end