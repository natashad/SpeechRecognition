function [LL] = loglikHMM( HMM, data )
% trainHMM
%
%  inputs:
%          HMM       : the HMM in which to compute the likelihood
%          data(d,n) : Matrix, d by n.
%
%     outputs:
%          LL           : the log likelihood of the data given the HMM
%
%  (c) Frank Rudzicz 2010

engine = smoother_engine(jtree_2TBN_inf_engine(HMM));
evidence = cell( size(HMM.intra,1), size(data,2) );
evidence(HMM.observed,:) = num2cell( data, 1);

[engine, LL] = enter_evidence( engine, evidence );

return