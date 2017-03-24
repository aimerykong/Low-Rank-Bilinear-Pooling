function pre = addFroNormAct_backward(layer, pre, now)
%
%
%
% Shu Kong @ UCI
% Oct, 2016

%% fetch weights, bias, param
W = layer.weights{1}; % reducedDim(512) x r*C (20dim*200classes)
b = layer.weights{2}; % 1 x C
r = layer.rDim;
C = layer.nClass;
lambda = layer.lambda;

[~, rC] = size(W);
assert(rC == r*C, 'incorrect weights at the biCls layer!');
%{
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
%}
%% gradient on x
preX = gather(pre.x);  % rC x h*w*N
now_dzdx = squeeze(gather(now.dzdx)); % dzdy
out = 2*W*now_dzdx;
out = reshape(out', [28,28,size(out,1)]);

%% return
outw = zeros(size(W), 'single');
outb = zeros(size(b),'single');
pre.dzdw = {outw, outb};

pre.dzdx = gpuArray(single(out));



