function [HMM,LL] = trainHMM( HMM, data, max_iter ) 
% [HMM,LL] = trainHMM( HMM, data, max_iter ) 
%
%    inputs:
%          HMM          : the HMM we wish to train
%          data{i}(d,n) : Cell array. The i'th sequence has dimension d and
%                         length n
%          max_iter     : the maximum iterations of EM
%
%     outputs:
%          HMM          : the HMM object, with updated parameters
%          LL           : the log likelihood of the training data on the
%                         updated HMM
%
%  (c) Frank Rudzicz 2010


if nargin < 3
    max_iter = 20;
end

if isempty(HMM)
    disp('Empty HMM. Skipping.');
    LL = -inf;
    return
end

% training
engine = smoother_engine(jtree_2TBN_inf_engine(HMM));
evidence = cell(1, size( data, 2));
for l=1:size(evidence,2)
    evidence{l} = cell( size(HMM.intra,1), size(data{l},2));
    evidence{l}(HMM.observed,:) = num2cell(data{l},1);
end

[HMM, LL] = learn_params_dbn_em( engine, evidence, 'max_iter', max_iter);

return
