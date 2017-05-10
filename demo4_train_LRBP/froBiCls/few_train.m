function trainset=few_train(imdb, k)
% random select k samples from each training class
    rng('default');
    nclass=max(imdb.images.label);
    
    trainset=[];
    for i=1:nclass
        cur=((imdb.images.set==1) & (imdb.images.label==i));
        cur=find(cur);
        trainset=[trainset cur(randsample(numel(cur), k))];
    end
end