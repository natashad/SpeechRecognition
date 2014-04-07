function gmms = gmmTrain( dir_train, max_iter, epsilon, M )
% gmmTain
%
%  inputs:  dir_train  : a string pointing to the high-level
%                        directory containing each speaker directory
%           max_iter   : maximum number of training iterations (integer)
%           epsilon    : minimum improvement for iteration (float)
%           M          : number of Gaussians/mixture (integer)
%
%  output:  gmms       : a 1xN cell array. The i^th element is a structure
%                        with this structure:
%                            gmm.name    : string - the name of the speaker
%                            gmm.weights : 1xM vector of GMM weights
%                            gmm.means   : DxM matrix of means (each column
%                                          is a vector
%                            gmm.cov     : DxDxM matrix of covariances.
%                                          (:,:,i) is for i^th mixture

DD = dir(dir_train);

gmms = {};

% Starting at 3 skips '.' and '..'
for folder=3:length(DD)
    % Initialize theta

    index = folder-2;

    path = [dir_train, filesep, DD(folder).name, filesep];
    D2 = dir( horzcat(path, ['*', 'mfcc']) );

    mfccMatrix = [];

    for iFile=1:length(D2)
        mfccMatrix = vertcat(mfccMatrix, load(horzcat(path,[D2(iFile).name])));
    end

    [mfccMatrix, eigVecs, pcameans] = pca(mfccMatrix, 10);

    d = size(mfccMatrix,2);
    omega = [];

    for m=1:M
        omega(:,:,m) = eye(d);
    end

    k = randperm(size(mfccMatrix,1));
    mus = mfccMatrix(k(1:M),:);
    mus = mus';

    weights = ones(1,M)*(1/M);

    % theta initialization complete.

    i = 0;
    prev_L = -Inf;
    improvement = Inf;
    T = size(mfccMatrix,1);

    while i <= max_iter & improvement >= epsilon
        [weights, mus, omega, L] = emStep(weights, mus, omega, mfccMatrix, M, T, d);
        improvement = L - prev_L;
        prev_L = L;
        i = i+1;
    end

    gmms{index} = {};
    gmms{index}.name = DD(folder).name;
    gmms{index}.weights = weights;
    gmms{index}.means = mus;
    gmms{index}.cov = omega;
    gmms{index}.pcaproject = eigVecs;
    gmms{index}.pcameans = pcameans;

end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Support functions           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [w, u, c, L]= emStep(w, u, c, x, M, T, D)

    b = zeros(T,M);

    for m=1:M
        um = u(:, m)'; %1xD
        cm = diag(c(:,:,m))'; %1xD
        numer = sum((((x-repmat(um, T, 1)).^2)./repmat(cm,T,1)), 2);
        numer = exp(-0.5 * numer); %Tx1

        denom = ((2*pi)^(D/2) * sqrt(prod(cm))); %scalar
        b(:,m) = numer/denom;
    end

    % b should now be calculated.

    p_x_theta = sum(repmat(w, T, 1).*b, 2); %Tx1
    L = sum(log2(p_x_theta));

    p_m_t = zeros(T,M); %TxM

    for m=1:M
        wm = w(1,m); %scalar
        p_m_t(:, m) = repmat(wm, T, 1).*b(:,m)./p_x_theta;
    end

    % Update parameters
    w = sum(p_m_t, 1)/T; %1xM
    u = (p_m_t' * x)'./repmat(sum(p_m_t, 1),D,1); %DxM
    c_d_m = (p_m_t' * (x.^2))'./repmat(sum(p_m_t, 1),D,1) - u.^2; %DxM

    for m=1:M
        c(:,:,m) = diag(c_d_m(:,m));
    end

end
