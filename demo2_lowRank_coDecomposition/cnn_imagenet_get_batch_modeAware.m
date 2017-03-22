function imo = cnn_imagenet_get_batch_modeAware(images, mode, varargin)
% CNN_IMAGENET_GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.imageSize = [227, 227] ;
opts.border = [29, 29] ;
opts.keepAspect = true ;
opts.numAugments = 1 ; % flip?
opts.transformation = 'none' ;  % 'stretch' 'none'
opts.averageImage = [] ;
opts.rgbVariance = 1*ones(1,1,'single') ; % default: zeros(0,3,'single') ;
% opts.rgbVariance = zeros(0,3,'single') ;

opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts = vl_argparse(opts, varargin);

% fetch is true if images is a list of filenames (instead of
% a cell array of images)
fetch = numel(images) >= 1 && ischar(images{1}) ;

% prefetch is used to load images in a separate thread
prefetch = fetch & opts.prefetch ;

if prefetch
    vl_imreadjpeg(images, 'numThreads', opts.numThreads, 'prefetch') ;
    imo = [] ;
    return ;
end
if fetch
    im = vl_imreadjpeg(images,'numThreads', opts.numThreads) ;
else
    im = images ;
end

tfs = [] ;
switch opts.transformation
    case 'none'
        tfs = [
            .5 ;
            .5 ;
            0 ] ;
    case 'f5'
        tfs = [...
            .5 0 0 1 1 .5 0 0 1 1 ;
            .5 0 1 0 1 .5 0 1 0 1 ;
            0 0 0 0 0  1 1 1 1 1] ;
    case 'f25'
        [tx,ty] = meshgrid(linspace(0,1,5)) ;
        tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
        tfs_ = tfs ;
        tfs_(3,:) = 1 ;
        tfs = [tfs,tfs_] ;
    case 'stretch'
    otherwise
        error('Uknown transformations %s', opts.transformation) ;
end
[~, transformations] = sort(rand(size(tfs,2), numel(images)), 1) ;

if ~isempty(opts.rgbVariance) && isempty(opts.averageImage)
    opts.averageImage = zeros(1,1,3) ;
end
if numel(opts.averageImage) == 3
    opts.averageImage = reshape(opts.averageImage, 1,1,3) ;
end

imo = zeros(opts.imageSize(1), opts.imageSize(2), 3, ...
    numel(images)*opts.numAugments, 'single') ;

si = 1 ;
% augment testset or not
% augTest_scale = false;
augTest_rotation = false; % false true
augTest_shift = false;
augTest_crop = false;
augTest_flip = false;
augTest_colorPCA = false;
for i=1:numel(images)
    %% acquire image
    if isempty(im{i})
        imt = imread(images{i}) ;
        imt = single(imt) ; % faster than im2single (and multiplies by 255)
    else
        imt = im{i} ;
    end
    if size(imt,3) == 1
        imt = cat(3, imt, imt, imt) ;
    end
    %% rand-scale    
    %{
    if  strcmp(mode, 'train')
        sz = size(imt);
        scaleFactor = 1+randn(1)*0.15;
        if scaleFactor>1.3
            scaleFactor=1.3;
        elseif scaleFactor<0.7
            scaleFactor=0.7;
        end            
        imt = imresize(imt, scaleFactor);
        newsz = size(imt);
        imt = imt( round((newsz(1)-sz(1))/2) );
    elseif strcmp(mode, 'test') && augTest_scale       
        scaleFactor = 1+randn(1)*0.1;
        if scaleFactor>1.1
            scaleFactor=1.1;
        elseif scaleFactor<0.9
            scaleFactor=0.9;
        end            
        imt = imresize(imt, scaleFactor);
    end
    %}
    %% rand-rotate
    if  strcmp(mode, 'train')
        rangeDegree = 7;
        angle = rangeDegree*randn(1,1)/4;
        if angle>rangeDegree
            angle = rangeDegree;
        end
        if angle<-1*rangeDegree
            angle = -1*rangeDegree;
        end
        W = size(imt,2);
        H = size(imt,1);
        rotImg = imrotate(imt, angle, 'bicubic');
        Hst = ceil(W*abs(sin(angle/180*pi)));
        Wst = ceil(H*abs(sin(angle/180*pi)));
        imt = rotImg(Hst:end-Hst, Wst:end-Wst, :);
    elseif (strcmp(mode,'test') || strcmp(mode,'val')) && augTest_rotation
        rangeDegree = 3;
        angle = rangeDegree*randn(1,1)/4;
        if angle>rangeDegree
            angle = rangeDegree;
        end
        if angle<-1*rangeDegree
            angle = -1*rangeDegree;
        end
        W = size(imt,2);
        H = size(imt,1);
        rotImg = imrotate(imt, angle, 'bicubic');
        Hst = ceil(W*abs(sin(angle/180*pi)));
        Wst = ceil(H*abs(sin(angle/180*pi)));
        imt = rotImg(Hst:end-Hst, Wst:end-Wst, :);        
    end       
    %% rand-shift by rand-crop
    if  strcmp(mode, 'train')
        sFactor = 0.07;
        w = size(imt,2);
        h = size(imt,1);
        leftPoint = randi(round(sFactor*w), 1, 1);
        topPoint = randi(round(sFactor*h), 1, 1);
        rightPoint = randi(floor(sFactor*h), 1, 1)-1;
        bottomPoint = randi(floor(sFactor*h), 1, 1)-1;
        imt = imt(topPoint:end-bottomPoint,leftPoint:end-rightPoint,:);    
    elseif  (strcmp(mode,'test') || strcmp(mode,'val')) && augTest_shift     
        sFactor = 0.04;
        w = size(imt,2);
        h = size(imt,1);
        leftPoint = randi(round(sFactor*w), 1, 1);
        topPoint = randi(round(sFactor*h), 1, 1);
        rightPoint = randi(floor(sFactor*h), 1, 1)-1;
        bottomPoint = randi(floor(sFactor*h), 1, 1)-1;
        imt = imt(topPoint:end-bottomPoint,leftPoint:end-rightPoint,:);
    end
    %% resize
    w = size(imt,2) ;
    h = size(imt,1) ;
    factor = [(opts.imageSize(1)+opts.border(1))/h ...
        (opts.imageSize(2)+opts.border(2))/w];
    
    if opts.keepAspect
        factor = max(factor) ;
    end
    if any(abs(factor - 1) > 0.0001)
        imt = imresize(imt, ...
            'scale', factor, ...
            'method', opts.interpolation) ;
    end   
    %% crop & flip
    w = size(imt,2) ;
    h = size(imt,1) ;
    for ai = 1:opts.numAugments
        switch opts.transformation
            case 'stretch'
                sz = round(min(opts.imageSize(1:2)' .* (1-0.1+0.2*rand(2,1)), [h;w])) ;
                dx = randi(w - sz(2) + 1, 1) ;
                dy = randi(h - sz(1) + 1, 1) ;
                if  strcmp(mode, 'train')
                    flip = rand(1) > 0.5; % random flip as data augmentation
                elseif strcmp(mode, 'test') && augTest_flip
                    flip = rand(1) > 0.5; % random flip as data augmentation
                else
                    flip = 0;
                end
            otherwise
                tf = tfs(:, transformations(mod(ai-1, numel(transformations)) + 1)) ;
                sz = opts.imageSize(1:2) ;
               
                dx = floor((w - sz(2)) * tf(2)) + 1 ;
                dy = floor((h - sz(1)) * tf(1)) + 1 ;
                %flip = tf(3) ;
                if  strcmp(mode, 'train')
                    flip = rand(1) > 0.5; % random flip as data augmentation
                    % flip = 0; % random flip as data augmentation
                    % flip = 1; % random flip as data augmentation
                elseif  (strcmp(mode,'test') || strcmp(mode,'val'))  && augTest_flip
                    % flip = 1; % random flip as data augmentation
                    % flip = 0; % do not flip for testing/validation data
                    flip = rand(1) > 0.5; % random flip as data augmentation
                else
                    flip = 0;
                end
        end
        sx = round(linspace(dx, sz(2)+dx-1, opts.imageSize(2))) ;
        sy = round(linspace(dy, sz(1)+dy-1, opts.imageSize(1))) ;
        %% rand crop
        if  strcmp(mode, 'train') % rand crop
            x_minusSth = sx(1)-1;
            x_plusSth = w-sx(end);
            y_minusSth = sy(1)-1;
            y_plusSth = h-sy(end);
            if x_minusSth>1
                sx = sx + randi(x_plusSth,1);
                sx = sx - randi(x_minusSth,1);
            end
            if y_minusSth>1
                sy = sy+ randi(y_plusSth,1);
                sy = sy - randi(y_minusSth,1);
            end
        elseif  (strcmp(mode, 'test') || strcmp(mode,'val')) && augTest_crop% rand crop
            x_minusSth = sx(1)-1;
            x_plusSth = w-sx(end);
            y_minusSth = sy(1)-1;
            y_plusSth = h-sy(end);
            if x_minusSth>1
                sx = sx + randi(x_plusSth,1);
                sx = sx - randi(x_minusSth,1);
            end
            if y_minusSth>1
                sy = sy+ randi(y_plusSth,1);
                sy = sy - randi(y_minusSth,1);
            end
        end
        %% rand flip
        if flip
            sx = fliplr(sx) ;
        end
        %% fancy pca--color jittering
        if ~isempty(opts.averageImage)
            offset = opts.averageImage ;
            if ~isempty(opts.rgbVariance)                
                if strcmp(mode, 'train')
                    offset = bsxfun(@plus, offset, reshape(opts.rgbVariance * randn(3,1)*1.0, 1,1,3)) ;
                elseif  (strcmp(mode,'test') || strcmp(mode,'val')) && augTest_colorPCA
                    offset = bsxfun(@plus, offset, reshape(opts.rgbVariance * randn(3,1)*0.2, 1,1,3)) ;
                end
            end
            
            % save to disk as jpeg format?
            %A = imt(sy,sx,:);
            %imwrite(uint8(A), 'tmp.jpg');
            %A = imread('tmp.jpg');
            %A = single(A);
            %imt(sy,sx,:) = A;
            
            imo(:,:,:,si) = bsxfun(@minus, imt(sy,sx,:), offset) ;
        else
            imo(:,:,:,si) = imt(sy,sx,:) ;
        end
        si = si + 1 ;
    end
end
