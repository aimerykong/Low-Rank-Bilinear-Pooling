function [U, b, objList, trAccList] = myFroBiSVMtrain(trainFs, y, rDimBiCls, ...
    reg_coeff, epsilon, MAXITER, lr, lambda, U, b)
%
%
%
%
%
% Shu Kong 
% skong2@uci.edu
% CS, UCI

%% configurations


trainFs = reshape(trainFs, [size(trainFs,1)^0.5, size(trainFs,1)^0.5, size(trainFs,2)]);

if nargin < 3
    rDimBiCls = round(size(trainFs,1)/20); % default learning rate
end
if nargin < 4
    reg_coeff = 0.001; % default learning rate
end
if nargin < 5
    epsilon = 0.001; % default learning rate
end
if nargin < 6
    MAXITER = 1000; % default learning rate
end
if nargin < 7
    lr = 0.001; % default learning rate
end
if nargin < 8
    lambda = 0.001; % default lambda -- weight for the L2 regularizer
end
if nargin < 9
    %sc = sqrt(2/(size(trainFs,1)*rDimBiCls));
    U = randn(size(trainFs,1), rDimBiCls, 'single'); % *sc 
end
if nargin < 10
    b = 0;
end
%% initialization
objList = zeros(1, MAXITER);
trAccList = zeros(3, MAXITER);   

%% first pass to get the objective function value
W = U*U'; % get the full projector to speed up forward-pass computation


traceList = W(:)'* reshape(trainFs, [numel(trainFs(:,:,1)), size(trainFs,3)]);

traceList = traceList + b;
signTraceList = y(:)' .* traceList; 
selectedFlag = zeros(1,length(signTraceList));
selectedFlag(signTraceList<1) = 1;

curObjVal = lambda/2*sum(U(:).^2) + 1/length(selectedFlag) * sum( (1-signTraceList(selectedFlag==1)) );
% objList(curIter) = curObjVal;

%% accuracy on initial W=U*U'
predTMP = traceList;
predTMP(traceList>0) = 1;
predTMP(traceList<=0) = -1;
% trAccList(1,curIter) = mean(y==predTMP); % overall accuracy
% trAccList(2,curIter) = mean(predTMP(y==1)==1); % precision
% trAccList(3,curIter) = mean(predTMP(y==-1)==-1); %

%% gradient on initial W=U*U'
validIdx = find(selectedFlag==1);
gradU = lambda*U;
gradb = 0;

if ~isempty(validIdx)    
    A = reshape(-1*y(validIdx), [1,1,length(validIdx)]);
    A = bsxfun(@times, A, trainFs(:,:,validIdx) );
    
%     A = repmat(A, [size(trainFs,1),size(trainFs,2),1]);
%     A = A.* trainFs(:,:,validIdx) ;
    A = sum(A,3);
    
    gradU =  gradU + 1/length(selectedFlag) * A*U;    
    gradb = gradb +  1/length(selectedFlag) * sum(-y(validIdx));
end

fprintf('\titer-0000, fval=%.9f [init], (accOverall=%.3f, PosAcc=%.3f, NegAcc=%.3f)\n', ...
        curObjVal, mean(y(:)==predTMP(:)), mean(predTMP(y==1)==1), mean(predTMP(y==-1)==-1) );

%% initialize&update
Uold = U;
bold = b;
curIter = 0;
curObjVal = Inf;

while curIter <= MAXITER && curObjVal > epsilon
    curIter = curIter + 1;
    U = Uold;
    b = bold;
    
    %% forward pass to get the objective function value
    W = U*U'; % get the full projector to speed up forward-pass computation
    
    traceList = W(:)'* reshape(trainFs, [numel(trainFs(:,:,1)), size(trainFs,3)]);
    traceList = traceList + b;
    signTraceList = y(:)' .* traceList;
    selectedFlag = zeros(1,length(signTraceList));
    selectedFlag(signTraceList<1) = 1;
    
    curObjVal = lambda/2*sum(U(:).^2) + 1/length(selectedFlag) * sum( (1-signTraceList(selectedFlag==1)) );
    objList(curIter) = curObjVal;
    
    %% accuracy (W=U*U')
    predTMP = traceList;
    predTMP(traceList>0) = 1;
    predTMP(traceList<=0) = -1;
    trAccList(1,curIter) = mean(y(:)==predTMP(:)); % overall accuracy
    trAccList(2,curIter) = mean(predTMP(y==1)==1); % precision
    trAccList(3,curIter) = mean(predTMP(y==-1)==-1); %

    %% gradient (W=U*U')
    validIdx = find(selectedFlag==1);
    gradU = lambda*U;
    gradb = 0;
    
    if ~isempty(validIdx)
        
        cury = y(validIdx);
        cury(cury==1) = 100;
        A = reshape(-1*cury, [1,1,length(validIdx)]);        
%         A = reshape(-1*y(validIdx), [1,1,length(validIdx)]);
        A = bsxfun(@times, A, trainFs(:,:,validIdx) );
%         A = repmat(A, [size(trainFs,1),size(trainFs,2),1]);
%         A = A.* trainFs(:,:,validIdx) ;
        A = sum(A,3);
        
%         gradU =  gradU + A*U;
%         gradb = gradb +  mean(-y(validIdx));
        
        gradU =  gradU + 1/length(selectedFlag) * A*U;    
        gradb = gradb +  1/length(selectedFlag) * sum(-y(validIdx));
    end
    
    fprintf('\titer-%04d, fval=%.9f, (accOverall=%.3f, PosAcc=%.3f, NegAcc=%.3f)',...
                curIter, objList(curIter), trAccList(1,curIter), trAccList(2,curIter), trAccList(3,curIter));
    fprintf('\tupdating...\n');
    %% update
    U = U - lr*gradU;
    b = b - lr*gradb;
    
    Uold = U;
    bold = b;
    
%     if curIter == 15
%         lr = 10*lr;
%     end
    
end

















