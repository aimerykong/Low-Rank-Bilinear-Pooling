function pre = bisvm_posneg_UUreg_backward(layer, pre, now)
%
% compute the gradient w.r.t weights and input data at this layer for
% backpropogation.
%
%
%
% Shu Kong @ UCI
% Sep, 2016

%% fetch weights, bias, param
updatePos = true; updateNeg = ~updatePos;% update positive part only
% updatePos = false; updateNeg = ~updatePos; % update negative part only
% updatePos = true; updateNeg = updatePos;% update both parts


W = layer.weights{1}; % reducedDim(512) x r*C (20dim*200classes)
b = layer.weights{2}; % 1 x C
r = layer.rDim;
C = layer.nClass;
lambda = layer.lambda;

[~, rC] = size(W);
assert(rC == r*C, 'incorrect weights at the biCls layer!');

%% space for computational efficiency
Wtensor = reshape(W, [size(W,1), r, C]);
UUpos = cell(1,C);
gramSumUUpos = 0;
UUneg = cell(1,C);
gramSumUUneg = 0;

for cidx = 1:C
%     curU = Wtensor(:,:,cidx); % 512xr
    curUpos = Wtensor(:,1:r/2,cidx); % 512x(r/2)
    curUneg = Wtensor(:,1+r/2:end,cidx); % 512x(r/2)
    
    UUpos{cidx} = curUpos*curUpos';
    gramSumUUpos = gramSumUUpos + UUpos{cidx};
    UUneg{cidx} = curUneg*curUneg';
    gramSumUUneg = gramSumUUneg + UUneg{cidx};
    
%     curU = W(:,1+(cidx-1)*r:r*cidx);
%     UU{cidx} = curU*curU';
%     gramSumUU = gramSumUU + UU{cidx};
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
        curOut =  2*curDer(cidx)*(UUpos{cidx}-UUneg{cidx});    
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
    outWtmp = 0;
    outbtmp = 0;
    curUpos = Wtensor(:,1:r/2,cidx); % 512x(r/2)
    curUneg = Wtensor(:,1+r/2:end,cidx); % 512x(r/2)
    for i = 1:N
        curDer = now_dzdx(:,i);
        outWtmp = outWtmp + 2*curDer(cidx)*XX{i};
        outbtmp = outbtmp + curDer(cidx);       
        outb(cidx) = outb(cidx) + curDer(cidx); 
    end
    if updatePos
        outWpos = outWtmp*curUpos + 4*lambda*(curUpos*(curUpos'*curUpos)) + 4*lambda*(curUpos*(curUneg'*curUneg));
    else
        outWpos = zeros(size(outWtmp,1), size(curUpos,2), 'single');
    end
    if updateNeg
        outWneg = -outWtmp*curUneg + 4*lambda*(curUneg*(curUneg'*curUneg)) + 4*lambda*(curUneg*(curUpos'*curUpos));
    else
        outWneg = zeros(size(outWtmp,1), size(curUneg,2), 'single');
    end
    outw{cidx} = [outWpos,outWneg];
    
%     curU = W(:,1+(cidx-1)*r:r*cidx);
%     for i = 1:N
%         curDer = now_dzdx(:,i);
%         outw{cidx} = outw{cidx} + 2*curDer(cidx)*XX{i};
%         outb(cidx) = outb(cidx) + curDer(cidx);
%     end
%     outw{cidx} = outw{cidx}*curU + lambda*(curU*(curU'*curU));
end

%% return
outw = cell2mat(outw);
pre.dzdw = {outw, outb};
pre.dzdx = out;


%% gpu version
%{
function pre = bisvm_posneg_UUreg_backward(layer, pre, now)
%
% compute the gradient w.r.t weights and input data at this layer for
% backpropogation.
%
%
%
% Shu Kong @ UCI
% Sep, 2016

gpuMode = isa(pre.x, 'gpuArray');
%% fetch weights, bias, param
W = layer.weights{1}; % reducedDim(512) x r*C (20dim*200classes)
b = layer.weights{2}; % 1 x C
r = layer.rDim;
C = layer.nClass;
lambda = layer.lambda;

[~, rC] = size(W);
assert(rC==r*C, 'incorrect weights at the biCls layer!');

%% space for computational efficiency
Wtensor = reshape(W, [size(W,1), r, C]);
UUpos = cell(1,C);
UUneg = cell(1,C);
if gpuMode
    gramSumUUpos = gpuArray(zeros(size(W,1),size(W,1),'single'));
    gramSumUUneg = gpuArray(zeros(size(W,1),size(W,1),'single'));
else
    gramSumUUpos = zeros(size(W,1),size(W,1),'single');
    gramSumUUneg = zeros(size(W,1),size(W,1),'single');
end

for cidx = 1:C
%     curU = Wtensor(:,:,cidx); % 512xr
    curUpos = Wtensor(:,1:r/2,cidx); % 512x(r/2)
    curUneg = Wtensor(:,1+r/2:end,cidx); % 512x(r/2)
    
    UUpos{cidx} = curUpos*curUpos';
    gramSumUUpos = gramSumUUpos + UUpos{cidx};
    UUneg{cidx} = curUneg*curUneg';
    gramSumUUneg = gramSumUUneg + UUneg{cidx};
    
%     curU = W(:,1+(cidx-1)*r:r*cidx);
%     UU{cidx} = curU*curU';
%     gramSumUU = gramSumUU + UU{cidx};
end


% [h,w,c,N] = size(pre.x{1});
[h,w,c,N] = size(pre.x);
if gpuMode
    preX = pre.x;
    now_dzdx = squeeze(now.dzdx); % dzdy
else
    % preX = gather(pre.x{1});
    preX = gather(pre.x);
    now_dzdx = squeeze(gather(now.dzdx)); % dzdy
end

XX = cell(1,N);
if gpuMode
    gramSumXX = gpuArray(zeros(c,c,'single'));
else
    gramSumXX = 0;
end

for i = 1:N
    curX = preX(:,:,:,i); % h x w x c x Nc
    curX = reshape(curX, [h*w, c] );
    XX{i} = curX'*curX;
    gramSumXX = gramSumXX + XX{i}; % c x hw
end

%% gradient on x
if gpuMode
    out = gpuArray(zeros([h, w, c, N], 'like', now.dzdx));
else
    out = zeros([h, w, c, N], 'like', now.dzdx);
end
for i = 1:N
    %curX = pre.x(:,:,:,i); % h x w x c x Nc
    curX = preX(:,:,:,i); % h x w x c x Nc
    curX = reshape(curX, [h*w, c] );
    curX = curX'; % c x hw
    curDer = now_dzdx(:,i); %C x 1, or (nClass x 1)
    
    outTMP = zeros(size(UUpos{cidx}),'single');
    for cidx = 1:C
        curOut =  2*curDer(cidx)*(UUpos{cidx}-UUneg{cidx});    
        outTMP = outTMP + curOut;
    end
    outTMP = outTMP * curX;
    outTMP = outTMP'; % hw x c
    outTMP = reshape(outTMP, [h, w, c]);
    out(:,:,:,i) = outTMP;
end

%% gradient on weights
outw = cell(1, C);
if gpuMode
    outb = gpuArray(zeros(1, C));
else
    outb = zeros(1, C);
end
for cidx = 1:C
    if gpuMode
        outw{cidx} = gpuArray(zeros(size(c, r),'single'));
        outWtmp = gpuArray(zeros(size(W,1),size(W,1),'single'));
    else
        outw{cidx} = zeros(size(c, r));
        outWtmp = zeros(size(W,1),size(W,1),'single');
    end
    
%     outbtmp = zeros(size(W,1),size(W,1),'single');
    curUpos = Wtensor(:,1:r/2,cidx); % 512x(r/2)
    curUneg = Wtensor(:,1+r/2:end,cidx); % 512x(r/2)
    for i = 1:N
        curDer = now_dzdx(:,i);
        outWtmp = outWtmp + 2*curDer(cidx)*XX{i};
%         outbtmp = outbtmp + curDer(cidx);
        outb(cidx) = outb(cidx) + curDer(cidx);
    end
    outWpos = outWtmp*curUpos + 4*lambda*(curUpos*(curUpos'*curUpos)) + 4*lambda*(curUpos*(curUneg'*curUneg)); 
    outWneg = -outWtmp*curUneg + 4*lambda*(curUneg*(curUneg'*curUneg)) + 4*lambda*(curUneg*(curUpos'*curUpos)); 
    outw{cidx} = gather([outWpos,outWneg]);
    
%     curU = W(:,1+(cidx-1)*r:r*cidx);
%     for i = 1:N
%         curDer = now_dzdx(:,i);
%         outw{cidx} = outw{cidx} + 2*curDer(cidx)*XX{i};
%         outb(cidx) = outb(cidx) + curDer(cidx);
%     end
%     outw{cidx} = outw{cidx}*curU + lambda*(curU*(curU'*curU));
end

%% return
% for i = 1:length(outw)
%     outw{i} = gather(outw{i});
% end
outw = gpuArray(cell2mat(outw));
pre.dzdw = {outw, outb};
pre.dzdx = out;
%}





