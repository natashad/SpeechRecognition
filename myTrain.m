dir_train = '/u/cs401/speechdata/Training';
M = 8;
Q = 3;
initType = 'kmeans';
fn_HMM = 'savedHMM.mat';
max_iter = 5;
D = 14;

DD = dir(dir_train);

phnData = {};
amountOfTrainingData = length(DD);


% starting at 3 skips '.' and '..'
for s=3:amountOfTrainingData

    path_ = [dir_train, filesep, DD(s).name, filesep];
    D2 = dir([path_, '*phn']);

    for phnFile=1:length(D2)

        [Starts, Ends, Phns] = textread([path_,  D2(phnFile).name], '%d %d %s', 'delimiter','\n');

        file_name = D2(phnFile).name;
        mfcc_filename = [file_name(1:end-3), 'mfcc'];

        X = load([path_, mfcc_filename]);
        X = X';
        X = X(1:D, :);

        for p = 1:length(Phns)

            Start = Starts(p)/128 + 1;
            End = min(Ends(p)/128 + 1, length(X));
            phn = char(Phns(p));
            if strcmp(phn, 'h#')
                phn = 'sil';
            end

            if ~isfield(phnData, phn)
                phnData.(phn) = {};
            end

            phnData.(phn){length(phnData.(phn))+1} = X(:, Start:End);

        end


    end

end

HMM = struct();

fields = fieldnames(phnData);

for i = 1:length(fields)

    phn = fields{i};

    HMM.(phn) = initHMM( phnData.(phn), M, Q );

    [HMM.(phn), LL] = trainHMM( HMM.(phn), phnData.(phn), max_iter );


end

save( fn_HMM, 'HMM', '-mat');
