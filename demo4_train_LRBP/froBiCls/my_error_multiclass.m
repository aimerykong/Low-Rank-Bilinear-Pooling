% -------------------------------------------------------------------------
function err = my_error_multiclass(opts, labels, res)
% -------------------------------------------------------------------------

predictions = gather(res(end-1).x) ;
if size(predictions,1) ~= 1 && size(predictions,2)~=1
    predictions = reshape(predictions, [1,1,size(predictions,1),size(predictions,2)]);
end
  
[~,predictions] = sort(predictions, 3, 'descend') ;

% be resilient to badly formatted labels
if numel(labels) == size(predictions, 4)
    labels = reshape(labels,1,1,1,[]) ;
end

% skip null labels
mass = single(labels(:,:,1,:) > 0) ;
if size(labels,3) == 2
    % if there is a second channel in labels, used it as weights
    mass = mass .* labels(:,:,2,:) ;
    labels(:,:,2,:) = [] ;
end

m = min(5, size(predictions,3)) ;

error = ~bsxfun(@eq, predictions, labels) ;
err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:m,:),[],3)))) ;


%{
predictions = gather(res(end-1).x) ;
[~, predictions] = sort(predictions, 1, 'descend') ;

% be resilient to badly formatted labels
if numel(labels) == size(predictions, 2)
    labels = reshape(labels,1,[]) ;
end

% skip null labels
mass = single(labels > 0) ;
if size(labels,1) == 2
    % if there is a second channel in labels, used it as weights
    mass = mass .* labels(:,:,2,:) ;
    labels(:,:,2,:) = [] ;
end

m = min(5, size(predictions,3)) ;

error = ~bsxfun(@eq, predictions, labels) ;
err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:m,:),[],3)))) ;
%}
