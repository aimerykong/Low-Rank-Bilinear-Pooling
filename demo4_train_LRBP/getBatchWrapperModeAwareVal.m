% return a get batch function
% -------------------------------------------------------------------------
function fn = getBatchWrapperModeAwareVal(opts)
% -------------------------------------------------------------------------
    fn = @(imdb,batch,mode) getBatchVal(imdb,batch, mode, opts) ;
end

% -------------------------------------------------------------------------
function [im,labels] = getBatchVal(imdb, batch, mode, opts)
% -------------------------------------------------------------------------
    images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
    im = cnn_imagenet_get_batch_modeAwareVal(images, mode, opts, ...
                                'prefetch', nargout == 0) ;
    labels = imdb.images.label(batch) ;
end
