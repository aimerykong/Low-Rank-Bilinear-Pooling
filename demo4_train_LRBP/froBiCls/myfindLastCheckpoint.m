% -------------------------------------------------------------------------
function epoch = myfindLastCheckpoint(modelDir, prefixStr)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, sprintf('%snet-epoch-*.mat', prefixStr))) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;


