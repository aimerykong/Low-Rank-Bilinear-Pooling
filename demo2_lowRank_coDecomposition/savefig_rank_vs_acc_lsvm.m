
rankAcc = load('rank_vs_acc_lsvm.mat', 'accList', 'rankList');

maxVisRank = 80;
figure;
plot(rankAcc.rankList(1:maxVisRank), rankAcc.accList(1:maxVisRank), 'r.-');
xlabel('rank of linear SVM');
ylabel('accuracy');


fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'rank_vs_acc_lsvm.pdf','-dpdf')