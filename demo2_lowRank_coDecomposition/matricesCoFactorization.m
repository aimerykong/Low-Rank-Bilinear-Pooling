function [U, P, Wapprox] = matricesCoFactorization(W, rankK, flag_optimization)
% It factorizes matcies {W1, ..., Wi, ..., Wn} with dimension dxm 
%	into a common matrix P with dimension dxr where r is the rankK
%   and (smaller) matrices {U1, ..., Ui, ..., Un} with dimension rxm
%   such that
%   min_{Ui,P} sum_i || W_i-P*Ui ||_fro^2, s.t. ||p_j||_2 = 1 for j=1,...,r
%
% It also returns Wapprox_i = P*Ui
%
% Shu Kong
% Sep. 2016
%

%% initialization
if ~exist('flag_optimization','var')
    flag_optimization = false;
end

[d, m, N] = size(W); % W in d x m x N, usually d>m
% Wapprox = zeros(d,m,N);
Wtilde = reshape(W, [d, m*N]);
SIGMAtmp = Wtilde*Wtilde';
[P,~,~] = svd(SIGMAtmp);
P = P(:,1:rankK);
% U = zeros(rankK, m, N);
U = P'*Wtilde;
Wapprox = P*U;
U = reshape(U, [rankK, m, N]);
Wapprox = reshape(Wapprox, [d, m, N]);

% for i = 1:N
%     U(:,:,i) = P'*W(:,:,i);
%     Wapprox(:,:,i) = P*U(:,:,i);
% end

if ~flag_optimization
    return;
end

%% optimization -- no need to optimize that the optimal is proven to be PCA initialization
MAXITER = 20;
flag_converge = false;
epsilon = 10e-8;
iter = 1;
diffListP = zeros(1, MAXITER);
diffListU = zeros(1, MAXITER);
diffListW = zeros(1, MAXITER);
P_pre = P;
U_pre = U;
Wapprox_pre = Wapprox;
while iter<MAXITER && ~flag_converge
    %% updating P
    Utilde = reshape(U, [rankK, m*N]);
    P = Wtilde*Utilde'/(Utilde*Utilde');
    Plength = sqrt(sum(P.^2,1));
%     validIdx = find(Plength>1);
%     P(:,validIdx) = P(:,validIdx) ./ repmat(Plength(validIdx), size(P,1), 1);
    P = P ./ repmat(Plength+epsilon, size(P,1), 1);
    
    %% updating Ui and Wapprox
    for i = 1:N
        U(:,:,i) = (P'*P)\P'*W(:,:,i);        
        Wapprox(:,:,i) = P*U(:,:,i);
    end    
    %% check convergenece
    diffListP(iter) = norm(P(:)-P_pre(:));
    diffListU(iter) = norm(U(:)-U_pre(:));
    diffListW(iter) = norm(Wapprox(:)-Wapprox_pre(:));
    fprintf('iter-%d diff-P:%.8f, diff-U:%.8f, diff-Wapprox:%.8f, \n', iter, diffListP(iter), diffListU(iter), diffListW(iter) );
    
    if diffListP(iter) < epsilon && diffListU(iter) < epsilon 
        flag_converge = true;
        fprintf('\nconverged at iter-%d\n', iter);
    end   
    
    P_pre = P;
    U_pre = U;
    iter = iter + 1;
end






