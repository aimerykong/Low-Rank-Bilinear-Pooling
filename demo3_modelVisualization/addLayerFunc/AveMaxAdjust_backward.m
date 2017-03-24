function pre = AveMaxAdjust_backward(layer, pre, now)
%
% compute the gradient w.r.t weights and input data at this layer for
% backpropogation.
%
%
%
% Shu Kong @ UCI
% Aug 17, 2016

%% fetch weights, bias, param
inX = pre.x;
alpha = layer.alpha;
% [h, w, c, N] = size(inX); % H  W  C  N
% inX = permute( inX, [3, 1, 2, 4] ); % size is h, w, c, n
% inX = reshape(inX, [c, h*w, N]);

% out = inX.^alpha;
%% space for computational efficiency
%{
UU = cell(1,C);
gramSumUU = 0;
for cidx = 1:C
    curU = W(:,1+(cidx-1)*r:r*cidx);
    UU{cidx} = curU*curU';
    gramSumUU = gramSumUU + UU{cidx};
end

% preX = gather(pre.x{1});
preX = gather(pre.x);
% [h,w,c,N] = size(pre.x{1});
[h,w,c,N] = size(pre.x);
now_dzdx = squeeze(gather(now.dzdx)); % dzdy
XX = cell(1,N);
gramSumXX = 0;
for i = 1:N
    curX = preX(:,:,:,i); % h x w x c x Nc
    curX = reshape(curX, [h*w, c] );
    XX{i} = curX'*curX;
    gramSumXX = gramSumXX + XX{i}; % c x hw
end

%% gradient on x
out = zeros([h, w, c, N], 'like', now.dzdx);
for i = 1:N
    %curX = pre.x(:,:,:,i); % h x w x c x Nc
    curX = preX(:,:,:,i); % h x w x c x Nc
    curX = reshape(curX, [h*w, c] );
    curX = curX'; % c x hw
    curDer = now_dzdx(:,i); %C x 1, or (nClass x 1)
    
    outTMP = 0;
    for cidx = 1:C
        %curU = W(:,1+(cidx-1)*r:r*cidx);
        %curOut =  2*curDer(cidx)*(curU*curU')*curX;
        curOut =  2*curDer(cidx)*UU{cidx};%*curX;
        %curOut = curOut'; % hw x c
        %curOut = reshape(curOut, [h, w, c]);
        outTMP = outTMP + curOut;
    end
    outTMP = outTMP * curX;
    outTMP = outTMP'; % hw x c
    outTMP = reshape(outTMP, [h, w, c]);
    out(:,:,:,i) = outTMP;
end

%% gradient on weights
outw = cell(1, C);
outb = zeros(1, C);
for cidx = 1:C
    outw{cidx} = zeros(size(c, r));
    curU = W(:,1+(cidx-1)*r:r*cidx);
    for i = 1:N
        curDer = now_dzdx(:,i);
        outw{cidx} = outw{cidx} + 2*curDer(cidx)*XX{i};
        outb(cidx) = outb(cidx) + curDer(cidx);
    end
    outw{cidx} = outw{cidx}*curU + lambda*(curU*(curU'*curU));
end
%}
%% slow version that tranverse all the data points
%{ 
    for cidx = 1:C
        outw{cidx} = zeros(size(c, r));
    end
    for i = 1:N
%         curX = pre.x(:,:,:,i); % h x w x c x N
        curX = preX(:,:,:,i); % h x w x c x Nc
        curX = reshape(curX, [h*w, c] );
        curX = curX'; % c x hw
        curDer = now_dzdx(:,i); %C x 1, or (nClass x 1)
        
        for cidx = 1:C
            curU = W(:,1+(cidx-1)*r:r*cidx);
            outw{cidx} = outw{cidx} + 2*curDer(cidx)*(curX*curX')*curU;
            outb(cidx) = outb(cidx) + curDer(cidx)*curDer(cidx);
        end
    end
%}
%% return
% outw = cell2mat(outw);
% pre.dzdw = {outw, outb};
% pre.dzdx = out;
pre.dzdx = now.dzdx *(1000/784);



