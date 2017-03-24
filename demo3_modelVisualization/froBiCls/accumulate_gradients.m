% -------------------------------------------------------------------------
function [state, net] = accumulate_gradients(state, net, res, opts, batchSize, mmap)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;

for l=numel(net.layers):-1:1
    for j=1:numel(res(l).dzdw)
        
        % accumualte gradients from multiple labs (GPUs) if needed
        if numGpus > 1
            tag = sprintf('l%d_%d',l,j) ;
            for g = otherGpus
                tmp = gpuArray(mmap.Data(g).(tag)) ;
                res(l).dzdw{j} = res(l).dzdw{j} + tmp ;
            end
        end
        
        if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
            % special case for learning bnorm moments
            thisLR = net.layers{l}.learningRate(j) ;
            net.layers{l}.weights{j} = ...
                (1 - thisLR) * net.layers{l}.weights{j} + ...
                (thisLR/batchSize) * res(l).dzdw{j} ;
        else
%             if l == 35
%                 disp(l);
%             end
            % standard gradient training
            thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
            thisLR = state.learningRate * net.layers{l}.learningRate(j) ;
            
            state.layers{l}.momentum{j} = opts.momentum * state.layers{l}.momentum{j} ...
                - thisDecay * net.layers{l}.weights{j} ...
                - (1 / batchSize) * res(l).dzdw{j} ;
            
            net.layers{l}.weights{j} = net.layers{l}.weights{j} + ...
                thisLR * state.layers{l}.momentum{j};
        end
        
        % if requested, collect some useful stats for debugging
        if opts.plotDiagnostics
            variation = [] ;
            label = '' ;
            switch net.layers{l}.type
                case {'conv','convt'}
                    variation = thisLR * mean(abs(state.layers{l}.momentum{j}(:))) ;
                    if j == 1 % fiters
                        base = mean(abs(net.layers{l}.weights{j}(:))) ;
                        label = 'filters' ;
                    else % biases
                        base = mean(abs(res(l+1).x(:))) ;
                        label = 'biases' ;
                    end
                    variation = variation / base ;
                    label = sprintf('%s_%s', net.layers{l}.name, label) ;
            end
            res(l).stats.variation(j) = variation ;
            res(l).stats.label{j} = label ;
        end
    end
end
