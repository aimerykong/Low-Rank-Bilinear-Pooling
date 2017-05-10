function out = imdb2set(set, isTrain)
% convert a set to indexes of train/test
    if isTrain
        out = find(set==1);
    else
        out = union(find(set==2), find(set==3));
    end
end