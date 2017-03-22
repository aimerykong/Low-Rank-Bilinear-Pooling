% return a get batch function
% -------------------------------------------------------------------------
function fn = getBatchWrapperModeAware(opts)
% -------------------------------------------------------------------------
    fn = @(imdb,batch,mode) getBatch(imdb,batch, mode, opts) ;
end

% -------------------------------------------------------------------------
function [im,labels] = getBatch(imdb, batch, mode, opts)
% -------------------------------------------------------------------------
    images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
    im = cnn_imagenet_get_batch_modeAware(images, mode, opts, ...
                                'prefetch', nargout == 0) ;
    labels = imdb.images.label(batch) ;
end
