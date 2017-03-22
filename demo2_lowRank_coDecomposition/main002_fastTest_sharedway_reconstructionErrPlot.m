% load('scorelist-04-Sep-201613:48:04_caffenoVal.mat');

orgDim = 512;
nclasses = 200;
flag_optimization = false;
rDimBiCls = 100; % reduced dimension by bilinear SVM
dimFixed = rDimBiCls/2;

W = reshape(W, [orgDim, orgDim, nclasses]);
%% svd for positive and negative parts
UList = zeros(orgDim, orgDim,nclasses);
VList = zeros(orgDim, orgDim,nclasses);
singularList = zeros(orgDim, nclasses);

posUList = zeros(orgDim, dimFixed, nclasses);
posSingularList = zeros(dimFixed, nclasses);
negUList = zeros(orgDim, dimFixed, nclasses);
negSingularList = zeros(dimFixed, nclasses);
W_lsvm = zeros(orgDim, orgDim, nclasses);
Wpos = zeros(orgDim,dimFixed,nclasses);
Wneg = zeros(orgDim,dimFixed,nclasses);

for c = 1:nclasses    
    curW = W(:,:,c);
    [curU, curD, curV] = svd( curW );        
    curd = diag(curD);    
    %% positive&negative
    sgnU = sign(curU(1,:));
    sgnV = sign(curV(1,:));
    a = sgnV.*sgnU;
    curd(a==-1) = curd(a==-1)*(-1);
    curd = curd(:);
    for j = 1:length(a)
        if a(j)==-1
            curV(:,j) = -1*curV(:,j);
        end
    end
    
    [dsort, didx] = sort(curd,'descend');
    UList(:,:,c) = curU(:,didx);
    VList(:,:,c) = curV(:,didx);
    singularList(:,c) = curd(didx);
    
    curU = UList(:,:,c);
    curd = singularList(:,c);
    posUList(:,:,c) = curU(:, 1:dimFixed);
    posSingularList(:,c) = curd(1:dimFixed);
    negUList(:,:,c) = curU(:, end-dimFixed+1:end);
    negSingularList(:,c) = curd( end-dimFixed+1:end);
%     
    Wpos(:,:,c) = posUList(:,:,c)*diag(posSingularList(:,c).^0.5);
    Wneg(:,:,c) = negUList(:,:,c)*diag((-1*negSingularList(:,c)).^0.5);
    %W_lsvm(:,:,c) = curW; % good
    W_lsvm(:,:,c) = Wpos(:,:,c)*Wpos(:,:,c)'-Wneg(:,:,c)*Wneg(:,:,c)'; % good
    %%
    if ~isempty(find(posSingularList(:,c)<-10e-7)) || ~isempty(find(negSingularList(:,c)>10e-7))             
        fprintf('attention class-%d -- ~isempty(find(posSingularList(:,i)<0)) || ~isempty(find(negSingularList(:,i)>0))\n', c);
    end
end
%% singular value
meanList = mean(singularList,2);
stdList = std(singularList');

figure(1);
maxLength = 2*30;
idx = [1:maxLength/2, length(meanList)+1-maxLength/2:length(meanList) ];
errorbar( 1:length(idx), meanList(idx)', stdList(idx)', '-o', 'MarkerSize',2, 'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'red' );
title('sorted eigen values with std for all classes');
xlabel('index of sorted singular value');
ylabel('eigen value'); 
grid on;

% figure(1);
% errorbar( 1:length(meanList), meanList(:)', stdList(:)', '-o', 'MarkerSize',2, 'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'red' );
% title('sorted eigen values with std for all classes');
% xlabel('index of sorted singular value');
% ylabel('eigen value'); 
% grid on;

%% rdim&rank --> reconstruct error
orgDim = 512;
%W2 = reshape(W, [orgDim,orgDim,nclasses]);
Wpos = zeros(orgDim,dimFixed,nclasses);
Wneg = zeros(orgDim,dimFixed,nclasses);
for i = 1:nclasses
    Wpos(:,:,i) = posUList(:,:,i)*diag(posSingularList(:,i).^0.5);
    Wneg(:,:,i) = negUList(:,:,i)*diag((-1*negSingularList(:,i)).^0.5);
%    W2(:,:,i) = Wpos(:,:,i)*Wpos(:,:,i)' - Wneg(:,:,i)*Wneg(:,:,i)'; 
end

dimRedList = 10:10:300; % m in P of size dxm
rankList = 2:2:20; % the total rank including positive and negative parts

% accMat = zeros(length(rankList),length(dimRedList));
% paramSizeMat = zeros(length(rankList),length(dimRedList));
reconErrMat = zeros(length(rankList),length(dimRedList));
curW = reshape(W_lsvm, [orgDim,orgDim,nclasses]);
curW = reshape(curW, [orgDim*orgDim, nclasses]);

for r = rankList
    WposTMP = Wpos(:,1:r/2,:);
    WnegTMP = Wneg(:,end-r/2+1:end,:);
    for m = dimRedList
%         if r > m
%             break;
%         end
        Wcat = cat(3, WposTMP,WnegTMP);
        [Ucat, P, Wcat_approx] = matricesCoFactorization(Wcat, m, flag_optimization);
        Wpos_approx = Wcat_approx(:,:,1:nclasses);
        Wneg_approx = Wcat_approx(:,:,1+nclasses:end);
        curW = zeros(orgDim,orgDim,nclasses);
        for i = 1:nclasses
            curW(:,:,i) = Wpos_approx(:,:,i)*Wpos_approx(:,:,i)' - Wneg_approx(:,:,i)*Wneg_approx(:,:,i)'; 
        end
        %% classification
%         curW = reshape(curW, [orgDim*orgDim, nclasses]);
%         score = bsxfun(@plus, curW'*valFV, b(:));
%         [~,predLabel] = max(score,[],1);
%         grndLabel = valY(:);
%         acc = mean(grndLabel(:)==predLabel(:));
%         rDim=512;m=512;r=20;nclasses=200;
%         paramSize = orgDim*m + nclasses*m*r;
%         paramSize = paramSize * 4/(1024^2);
        reconErr = norm(curW(:) - W(:),'fro');
%         fprintf('rank-%03d coDimRed-%03d on testset: acc=%.6f, paramSize: %.3f MB\n', r, m, acc, paramSize);
%         accMat(r/2,m/2) = acc; % accMat = zeros(length(rankList),length(dimRedList));
%         paramSizeMat(r/2,m/2) = paramSize;      
        reconErrMat(r/2,m/2) = reconErr;
    end
end

%% visualize reconstruction error
showMax_dim = 150;
withinMaxDimIdxs = find(dimRedList<=showMax_dim);
az = -60; % Azimuth
el = 20; % Elevation
nclasses = 200;

figure(2);
% subplot(1,2,2);
reconErrMatShow = reconErrMat ./ nclasses;
reconErrMatShow = reconErrMatShow(rankList/2,:);
reconErrMatShow = reconErrMatShow(:,dimRedList(withinMaxDimIdxs)/2);
bar3(reconErrMatShow);
xlabel('dimRed - m');
ylabel('rank - r');
zlabel('reconstruction error');
ax = gca;
ax.YTick = 1:length(rankList);
ax.YTickLabel = rankList;
ax.XTick = 1:length(dimRedList(withinMaxDimIdxs));
ax.XTickLabel = dimRedList(withinMaxDimIdxs);
title('reconstruction error vs. rank & dimRed');
view(az, el);

%% save
% save('reconErr_dimRed_rank_sharedDimParam4PosNeg.mat', 'reconErrMat', 'rankList', 'dimRedList',...
%     'singularList', 'meanList', 'stdList');



