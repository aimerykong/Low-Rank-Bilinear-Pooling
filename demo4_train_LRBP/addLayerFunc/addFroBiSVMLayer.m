function net = addFroBiSVMLayer( net, lossType, initFCparam, rDimBiCls, num_classes, learnW)
%
% add the bilinear SVM layer to a basemodel and a loss layer then.
%
%
% Shu Kong @ UCI
% June, 2016

%%
if sum(size(learnW)) == 2
    learnW = [learnW learnW*2];
    assert(sum(size(learnW)) ~= 2);
end

% convert the 'SVM' and 'LR' to internal representation
if strcmp(lossType, 'LR')
    lossType='softmaxlog';
elseif strcmp(lossType, 'SVM')
    lossType='hinge'; % mhinge
else
    error('unknown loss type');
end

net.layers{end+1} = struct('type', 'custom',...
    'forward', @biCls_forward, ...
    'backward', @biCls_backward, ...
    'name', 'biCls', ...
    'weights', initFCparam,... % 'learnW', learnW, 'learningRate', learnW, ...
    'weightDecay', [0.000001,0],...
    'rDim', rDimBiCls, 'nClass', num_classes);


% add the last softmax classification layer
%     net.layers{end+1} = struct('type', 'loss',...
%                              'name', 'softmaxLoss', ...
%                              'class', 'LR'); % lossType, class

net.layers{end+1} = struct('type', 'loss',...
    'name', 'final_loss', ...
    'class', lossType);


