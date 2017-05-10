function initFCparam = getFroBiSVMinitWeight(...
    initMethod, inputImgSize, nfeature, nclass,...
    classificationType, network, cfgId, dataset, netBeforeClassification,...
    use448, batchSize, rDimBiCls)
%
% Shu Kong @ UCI
% Jun. 2016

%% 
usePCA_onData = false;
usePCA_onWeights = ~usePCA_onData; % complementary

weight_file = fullfile('imdbFolder', dataset, 'exp', ['classifierW_' dataset '_' network '_' initMethod '_' classificationType '_' cfgId '_' num2str(inputImgSize) '.mat']);

if strcmp(initMethod, 'random')
    % random initialize
    %         initFCparam = {{init_weight('xavierimproved', 1, 1, nfeature, nclass, 'single'),...
    %                     zeros(nclass, 1, 'single')}};
    %         initFCparam = {{init_weight('xavierimproved', 1, nfeature(3), rDimBiCls, nclass, 'single'),...
    %                     zeros(nclass, 1, 'single')}};
    
    sc = sqrt(2/nclass) ; % xavierimproved
    weights = randn(nfeature(3), rDimBiCls, nclass, 'single')*sc ;
    initFCparam = {{weights, zeros(1, nclass, 'single')}};
    
elseif strcmp(initMethod, 'FroBiSVM') % using the Frobenius Bilinear SVM to pretrain the classifier
    if ~exist(weight_file, 'file') == 2
        % svm initialized weight, load from disk
        load(weight_file);
        
    elseif usePCA_onWeights % applying PCA on the weights from full linear SVM
        fprintf('compute the activations at the specific layer to pretrain the classifier...\n');
        [trainFV, trainY, valFV, valY] = get_activations_dataset_network_layer4FroBiSVM(...
            dataset, network, cfgId, classificationType, initMethod, use448, netBeforeClassification, batchSize);
        
        % train SVM or LR weight, and test it on the validation set.
        fprintf('pretrain the classifier...\n');
        [w, b, acc, map, scores] = train_test_myCode_vlfeat(classificationType, ...
            squeeze(trainFV), squeeze(trainY), squeeze(valFV), squeeze(valY));
        
        % reshape the parameters to the input format
        fprintf('SVD decomposition with hard low-rank constraint...\n');
        W = reshape(w, [size(w,1)^0.5 size(w,1)^0.5 size(w,2)] );
        U = zeros( [size(w,1)^0.5 rDimBiCls size(w,2)] );
        for i = 1:size(W,3)
            Wtmp = W(:,:,i);
            if norm(Wtmp-Wtmp','fro')~=0
                Wtmp = (Wtmp+Wtmp')/2.0;
            end
            [Ur, Sr] = svd(Wtmp);
            U(:,:,i) = Ur(:,1:rDimBiCls) * diag( diag(Sr(1:rDimBiCls, 1:rDimBiCls)).^0.5 );
        end
        
        U = single(U);
        b = single(squeeze(b));
        initFCparam = {{U, b}};
        
        % save on disk
        savefast(weight_file, 'initFCparam', 'U', 'b', 'acc', 'map', 'scores', 'w');
        
    elseif usePCA_onData % PCA reduction first on the data to expedite the pretraining process        
        fprintf('compute the activations at the specific layer to pretrain the classifier...\n');
        [trainFV, trainY, valFV, valY] = get_activations_dataset_network_layer4FroBiSVM(...
            dataset, network, cfgId, classificationType, initMethod, use448, netBeforeClassification, batchSize);        
                
%         [w, b, acc, map, scores] = train_test_myCode_vlfeat(classificationType, ...
%             squeeze(trainFV), squeeze(trainY), squeeze(valFV), squeeze(valY));
        
        
        % PCA on the data        

        trainFV = reshape(trainFV, [ size(trainFV,1)^0.5,  size(trainFV,1)^0.5, size(trainFV,2)]);
        valFV = reshape(valFV, [ size(valFV,1)^0.5,  size(valFV,1)^0.5, size(valFV,2)]); 
        meanData = mean(trainFV, 3);
%         trainFV = bsxfun(@minus, trainFV, meanData);
        gramMat = 0;
        for i = 1:size(trainFV,3)
            tmp = trainFV(:,:,i) - meanData;
            gramMat = gramMat + tmp*tmp';
        end
        gramMat = gramMat ./ size(trainFV,3);

%         gramMat = meanData;
        rDimBiCls = 20;
%         rDimBiCls = 40;
%         rDimBiCls = 60;
%         rDimBiCls = 120;
        [Ucommon, Scommon] = svd(gramMat);
        Ucommon = Ucommon(:,1:rDimBiCls);
        Scommon = diag(Scommon); 
        sum(Scommon(1:rDimBiCls))/sum(Scommon)
        %Scommon = Scommon(1:rDimBiCls);
        trainFV2 = zeros( rDimBiCls^2, size(trainFV,3) );
        valFV2 = zeros( rDimBiCls^2, size(valFV,3) );
        for i = 1:size(trainFV2,2)
            tmp = Ucommon'*trainFV(:,:,i)*Ucommon;
            trainFV2(:,i) = tmp(:);
        end
        %clear trainFV
        for i = 1:size(valFV2,2)
            tmp = Ucommon'*valFV(:,:,i)*Ucommon;
            valFV2(:,i) = tmp(:);
        end 
        %clear valFV
        trainFV2 = single(trainFV2);
        valFV2 = single(valFV2);
        % train SVM or LR weight, and test it on the validation set.
        fprintf('pretrain the classifier...\n');
        [w, b, acc, map, scores] = train_test_myCode_vlfeat(classificationType, ...
            squeeze(trainFV2), squeeze(trainY), squeeze(valFV2), squeeze(valY));
%         trainFV = reshape(trainFV, [size(trainFV,1)^2, size(trainFV,3)]);
%         valFV = reshape(valFV, [size(valFV,1)^2, size(valFV,3)]);

        % reshape the parameters to the input format
        fprintf('SVD decomposition with hard low-rank constraint...\n');
        W = reshape(w, [size(w,1)^0.5 size(w,1)^0.5 size(w,2)] );
        U = zeros( [size(trainFV,1) rDimBiCls size(w,2)] );
        for i = 1:size(W,3)
            Wtmp = W(:,:,i);
            if norm(Wtmp-Wtmp','fro')~=0
                Wtmp = (Wtmp+Wtmp')/2.0;
            end
            [Ur, Sr] = svd(Wtmp);
            Ur = Ur*diag(diag(Sr.^0.5));
            U(:,:,i) = Ucommon*Ur;
        end
        
        U = single(U);
        b = single(squeeze(b));
        initFCparam = {{U, b}};
        
        % save on disk
        savefast(weight_file, 'initFCparam', 'W', 'b', 'acc', 'map', 'scores');        
    else
        % TBA
        TBA = true; % :)
    end
    
elseif strcmp(initMethod, 'symmetricFullSVM') % using the standard full linear SVM to pretrain the classifier (symmetric input)
    %% symmetricFullSVM
    if exist(weight_file, 'file') == 2
        % svm or logistic initialized weight, load from disk
        load(weight_file);
    else
        % get activations from the last conv layer % checkpoint
%         [trainFV, trainY, valFV, valY] = get_activations_dataset_network_layer4symmetricFullSVM(...
%             dataset, network, cfgId, classificationType, initMethod, use448, netBeforeClassification, batchSize);
        
        [trainFV, trainY, valFV, valY] = get_activations_dataset_network_layer4FroBiSVM(...
            dataset, network, cfgId, classificationType, 'FroBiSVM', use448, netBeforeClassification, batchSize);   
        
        % train SVM or LR weight, and test it on the validation set.        
%         trainFV = reshape(trainFV, [size(trainFV,1)^2, size(trainFV,3)]);
%         valFV = reshape(valFV, [size(valFV,1)^2, size(valFV,3)]);       
        [w, b, acc, map, scores] = train_test_myCode_vlfeat(classificationType, ...
            squeeze(trainFV), squeeze(trainY), squeeze(valFV), squeeze(valY));
        
        % reshape the parameters to the input format
        w = reshape(single(w), 1, 1, size(w, 1), size(w, 2));
        b = single(squeeze(b));
        initFCparam = {{w, b}};
        
        % save on disk
        savefast(weight_file, 'initFCparam', 'acc', 'map', 'scores');        
    end
    
    %%%%%%% To be completed --- SVM dimensionality reduction, the other orthogonal matrix is for dimRed on conv5_-3, reshape initFCparam
    %     initFCparam
    initFCparam{1}{1} = squeeze(initFCparam{1}{1});
    initFCparam{1}{1} = reshape(initFCparam{1}{1}, [size(initFCparam{1}{1},1).^0.5, size(initFCparam{1}{1},1).^0.5, numel(initFCparam{1}{2}) ]);
    
    paramU = zeros(size(initFCparam{1}{1},1), rDimBiCls, size(initFCparam{1}{1},3));
    
    for i = 1:size(initFCparam{1}{1},3)
        tmp = initFCparam{1}{1}(:,:,i);
        if norm(tmp - tmp', 'fro')~=0
            tmp = (tmp+tmp')/2;
            initFCparam{1}{1}(:,:,i) = tmp;s
        end
        paramU(:,:,i) = initFCparam{1}{1}(:,1:rDimBiCls,i);
    end
    initFCparam{1}{1} = paramU;
    % end of using pretrain weight
else
    error('init method unknown');
end


