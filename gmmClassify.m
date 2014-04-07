%
% gmmClassify
%
% This is a script used to classify speakers.


dir_test = 'Testing';
dir_train = 'Training';
M = 8;
epsilon = 0.001;
fn_prefix = 'unkn_';
fn_suffix = 'lik';

gmms = gmmTrain(dir_train, 10, epsilon, M);

DD = dir([dir_test, filesep, '*', 'mfcc']);

for iFile=1:length(DD)

    DD(iFile).name;

    x1 = load([dir_test, filesep, DD(iFile).name]);
    T = size(x1, 1);
    D = size(x1,2);

    %log likelihood
    ll = zeros(length(DD),1);

    for s=1:length(gmms)
        u = gmms{s}.means;
        w = gmms{s}.weights;
        c = gmms{s}.cov;
        pcameans = gmms{s}.pcameans;
        pcameans = repmat(pcameans, T, 1);

        % re-projecting for pca
        eigVecs = gmms{s}.pcaproject;
        x = (x1-pcameans) * eigVecs;
        D = size(x,2);

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
        ll(s,1) = L;
    end

    indexing_mat = (1:length(ll))';
    ll2 = horzcat(indexing_mat, ll);
    ll2 = sortrows(ll2, 2);

    top1 = gmms{ll2(end,1)}.name;
    top2 = gmms{ll2(end-1,1)}.name;
    top3 = gmms{ll2(end-2,1)}.name;
    top4 = gmms{ll2(end-3,1)}.name;
    top5 = gmms{ll2(end-4,1)}.name;

    fname = DD(iFile).name(1:end-4);
    fname = [fname, fn_suffix];
    fID = fopen(fname, 'wt');
    fprintf(fID, '%s\n%s\n%s\n%s\n%s\n', top1, top2, top3, top4, top5);
    % fprintf(fID, '%s : %s\n', fname, top1);
    % temp = str2num(fname(end-5:end-4));
    % if (length(temp)==0 || temp < 16)
    %     disp([fname, ' : ' , top1]);
    % end
    fclose(fID);

end