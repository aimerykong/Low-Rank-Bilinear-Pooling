function initFCparam=getFCinitWeight(...
         initMethod,...
         nfeature, nclass,...
         classificationType, network, cfgId, dataset, netBeforeClassification,...
         use448, batchSize)
    
    if strcmp(initMethod, 'random')
        % random initialize
        initFCparam={{init_weight('xavierimproved', 1, 1, nfeature, nclass, 'single'),...
                    zeros(nclass, 1, 'single')}};
    elseif strcmp(initMethod, 'pretrain')
%         weight_file=consts(dataset, 'classifierW',...
%             'pretrainMethod', classificationType, ...
%             'network', network, ...
%             'cfgId', cfgId, ...
%             'projDim', nfeature, ...
%             'use448', use448);
        
%         weight_file = 'imdbFolder/CUB/exp/classifierW_CUB_VGG_M_LR_bilinear_224.mat';
        if use448
            weight_file = fullfile('imdbFolder', dataset, 'exp', ['classifierW_' dataset '_' network '_' classificationType '_' cfgId '_448.mat']);
        else
            weight_file = fullfile('imdbFolder', dataset, 'exp', ['classifierW_' dataset '_' network '_' classificationType '_' cfgId '_224.mat']);
        end
        
        if exist(weight_file, 'file') == 2
            % svm or logistic initialized weight, load from disk
            load(weight_file);
            %{
            if exist('w','var')                
                [w, b, acc, map, scores]= train_test_vlfeat(classificationType, ...
                    squeeze(trainFV), squeeze(trainY), squeeze(valFV), squeeze(valY));
                % reshape the parameters to the input format
                w=reshape(single(w), 1, 1, size(w, 1), size(w, 2));
                b=single(squeeze(b));
                initFCparam = {{w, b}};
            end
            %}
        else
            % get activations from the last conv layer % checkpoint
            [trainFV, trainY, valFV, valY]=...
                get_activations_dataset_network_layer(...
                    dataset, network, cfgId, use448, classificationType, initMethod, netBeforeClassification, batchSize);
            % train SVM or LR weight, and test it on the validation set. 
            [w, b, acc, map, scores]= train_test_vlfeat(classificationType, ...
                squeeze(trainFV), squeeze(trainY), squeeze(valFV), squeeze(valY));
            % reshape the parameters to the input format
            w=reshape(single(w), 1, 1, size(w, 1), size(w, 2));
            b=single(squeeze(b));
            initFCparam = {{w, b}};
            % save on disk
            savefast(weight_file, 'initFCparam', 'w', 'b', 'acc', 'map', 'scores');
        end
        % end of using pretrain weight
    else
        error('init method unknown');
    end
end