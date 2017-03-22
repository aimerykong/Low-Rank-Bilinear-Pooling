% load('scorelist-04-Sep-201613:48:04_caffenoVal.mat');
load reconErr_dimRed_rank_sharedDimParam4PosNeg.mat;
orgDim = 512;
nclasses = 200;
rDimBiCls = 100; % reduced dimension by bilinear SVM
dimFixed = rDimBiCls/2;

meanList = mean(singularList,2);
stdList = std(singularList');

figure(1);
maxLength = 2*30;
idx = [1:maxLength/2, length(meanList)+1-maxLength/2:length(meanList) ];
errorbar( 1:length(idx), meanList(idx)', stdList(idx)', '-o', 'MarkerSize',2, 'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'red' );
title('sorted eigen values with std for all classes');
xlabel('index of sorted eigen value');
ylabel('eigen value'); 
grid on;

% figure(1);
% errorbar( 1:length(meanList), meanList(:)', stdList(:)', '-o', 'MarkerSize',2, 'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'red' );
% title('sorted eigen values with std for all classes');
% xlabel('index of sorted singular value');
% ylabel('eigen value'); 
% grid on;

%% visualize reconstruction error
showMax_dim = 150;
withinMaxDimIdxs = find(dimRedList<=showMax_dim);
az = -60; % Azimuth
el = 20; % Elevation

figure(2);
% subplot(1,2,2);
reconErrMatShow = reconErrMat ./ (nclasses*orgDim*orgDim);
reconErrMatShow = reconErrMatShow(rankList/2,:);
reconErrMatShow = reconErrMatShow(:,dimRedList(withinMaxDimIdxs)/2);
bar3(reconErrMatShow);
xlabel('dimRed - m');
ylabel('rank - r');
zlabel('mean reconstruction error of entries');
ax = gca;
ax.YTick = 1:length(rankList);
ax.YTickLabel = rankList;
ax.XTick = 1:length(dimRedList(withinMaxDimIdxs));
ax.XTickLabel = dimRedList(withinMaxDimIdxs);
title('mean reconstruction error at entry level vs. rank & dimRed');
view(az, el);

%% save
% save('reconErr_dimRed_rank_sharedDimParam4PosNeg.mat', 'reconErrMat', 'rankList', 'dimRedList',...
%     'singularList', 'meanList', 'stdList');



