function [HMM] = initHMM( data, M, Q, initType )
% [HMM] = initHMM( data, M, Q, initType )
%
%  inputs:
%          data{i}(d,n) : Cell array. The i'th sequence has dimension d and
%                         length n
%          M            : the number of Gaussian mixture components
%                         per state (default 8)
%          Q            : the number of hidden states (default 3)
%          initType     : 'rnd' or 'kmeans' (default 'kmeans')
%
%  outputs:
%          HMM          : specialized HMM structure
%
%  (c) Frank Rudzicz 2010

nQ = 1;
nM = 2;
nO = 3;
intra = zeros(3);
intra(nQ,[nM nO]) = 1;
intra(nM,nO) = 1;
inter = zeros(3);
inter(nQ,nQ) = 1;

% default parameter sizes
if nargin < 4
    initType = 'kmeans';
end
if nargin < 3
    Q = 3;
end
if nargin < 2
    M = 8;
end
O = size(data{1},1); % size of observed vector

% create HMM
ns = [Q M O];
dnodes = [nQ nM];
onodes = [nO];
HMM = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'observed', onodes);

if isempty(data)
    disp('I need data');
    return;
end

% init probabilities
Qprior = zeros(Q,1); Qprior(1) = 1;
%Qtrans = [0.5 0.5 0; 0 0.5 0.5; 0.5 0 0.5];
%Qtrans = 0.5*eye(Q);
Qtrans = 0.5*eye(Q)+ diag(repmat(0.5,1,Q-1),1);
Qtrans(end,1) = 0.5
Qtrans = mk_stochastic( Qtrans );


mixmat = zeros(Q, M);
mu = zeros(O, Q, M);
sigma = zeros( O, O, Q, M);

if strcmp( initType, 'rnd')
    mixmat(:, :) = mk_stochastic( randn( Q, M ) );
    mu(:, :, :)  = mk_stochastic( randn( O, Q, M ) );
    sigma(:,:,:,:) = mk_stochastic( randn( O, O, Q, M));
elseif strcmp( initType, 'kmeans' )
    acousData = cell2mat(data);
    [gmmMu, gmmSig, gmmW] = mixgauss_init( M, acousData, 'full', initType );
    mixmat(:, :) = mk_stochastic( repmat(gmmW', Q, 1 ) );
    mu(:, :, : ) = reshape( repmat( gmmMu, Q, 1), [ O Q M]);
    sigma(:,:, :, :)  = reshape( repmat( gmmSig, 1, Q) , [ O O Q M ]);
else
    disp('If you specify the init type, use "rnd" or "kmeans"');
    HMM = [];
    return
end

% tabular CPDs
HMM.CPD{nQ}   = tabular_CPD(HMM, nQ, Qprior);
HMM.CPD{nM}   = tabular_CPD(HMM, nM, mixmat);
HMM.CPD{nO}   = gaussian_CPD(HMM, nO, 'mean', mu, 'cov', sigma);
HMM.CPD{nO+1} = tabular_CPD(HMM, nO+1, Qtrans);
return
