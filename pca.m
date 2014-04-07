function [pcadata, eigVecs, means] = pca(data, dim)

    D = size(data, 2);
    T = size(data, 1);

    % Centralize the data (subtract means)
    means = mean(data);
    meansRep = repmat(means, T, 1);
    data = data-meansRep;

    % Calculate the d x d covariance matrix
    cov_matrix = data' * data / (T-1);

    % % Calculate the eigenvectors of the covariance matrix
    % % Select m eigenvectors that correspond to the largest m eigenvalues to be the new basis.
    [eigVecs, eigVals] = eig(cov_matrix);
    [temp,eigenPositions]=sort(-eigVals);
    eigVecs = eigVecs(:,eigenPositions);
    eigVecs = eigVecs(:, 1:dim);

    pcadata = data * eigVecs;

end