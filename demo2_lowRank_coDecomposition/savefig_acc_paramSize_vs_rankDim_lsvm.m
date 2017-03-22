clc;

% quick setup
load('acc_dimRed_rank_sharedDimParam4PosNeg_bestTrainedModel.mat', ...
    'rankList', 'dimRedList', 'Wpos', 'Wneg',...
    'flag_optimization', 'paramSizeMat', 'accMat', 'grndLabel', 'b');


showMax_dim = 150;
withinMaxDimIdxs = find(dimRedList<=showMax_dim);
% withinMaxDimIdxs = withinMaxDimIdxs(1:end);
figure(2);
% subplot(1,2,1)
accMatShow = accMat;
accMatShow = accMatShow(rankList/2,:);
accMatShow = accMatShow(:,dimRedList(withinMaxDimIdxs)/2);
bar3(accMatShow);
xlabel('dimRed - m');
ylabel('rank - r');
zlabel('acc');
ax = gca;
ax.YTick = 1:length(rankList);
ax.YTickLabel = rankList;
ax.XTick = 1:length(dimRedList(withinMaxDimIdxs));
ax.XTickLabel = dimRedList(withinMaxDimIdxs);
title('acc vs. rank & dimRed');
az = -60; % Azimuth
el = 20; % Elevation
view(az, el);

figure(3);
% subplot(1,2,2);
paramSizeMatShow = paramSizeMat;
paramSizeMatShow = paramSizeMatShow(rankList/2,:);
paramSizeMatShow = paramSizeMatShow(:,dimRedList(withinMaxDimIdxs)/2);
bar3(paramSizeMatShow);
xlabel('dimRed - m');
ylabel('rank - r');
zlabel('size in MB');
ax = gca;
ax.YTick = 1:length(rankList);
ax.YTickLabel = rankList;
ax.XTick = 1:length(dimRedList(withinMaxDimIdxs));
ax.XTickLabel = dimRedList(withinMaxDimIdxs);
title('paramSize vs. rank & dimRed');
view(az, el);

[a,b] = max(accMatShow(:));
[y,x] = ind2sub(size(accMatShow),b);
fprintf('\nmax acc:%.4f, paramSize: %.3fMB, rank:%d in total, dimRed:%d\n', accMatShow(y,x), paramSizeMatShow(y,x), rankList(y), dimRedList(x));
disp(a)
disp(accMatShow(y,x)); % rankList(y), dimRedList(x)
disp( accMat(rankList(y)/2, dimRedList(x)/2 ) ); % rankList(y), dimRedList(x)

%% leave blank