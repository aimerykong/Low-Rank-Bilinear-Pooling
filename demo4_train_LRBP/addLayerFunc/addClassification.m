function net=addClassification(net, lossType, initFCparam)
    % convert the 'SVM' and 'LR' to internal representation
    if strcmp(lossType, 'LR')
        lossType='softmaxlog';
    elseif strcmp(lossType, 'SVM')
        lossType='mhinge';
    else
        error('unknown loss type');
    end

    net = addFC(net, 'fc_final', initFCparam, 'specified');
    
    % add the last softmax classification layer
    net.layers{end+1}=struct('type', 'loss',...
                             'name', 'final_loss', ...
                             'lossType', lossType);
end