dir_test = '/u/cs401/speechdata/Testing';
fn_output = 'speechrecog.txt';
fn_HMM = 'savedHMM.mat';

DMFCC = dir([dir_test, filesep, '*.mfcc']);
DPHN = dir([dir_test, filesep, '*.phn']);

fID = fopen(fn_output, 'w');
D = 14;

total = 0;
correct = 0;
wrong = 0;

load(fn_HMM);

for f=1:length(DPHN)
	fn_prefix = DPHN(f).name;
	fn_prefix = fn_prefix(1:end-3);
	fn_mfcc = [dir_test, filesep, fn_prefix, 'mfcc'];

	[starts, ends, phns] = textread([dir_test, filesep, DPHN(f).name],'%d %d %s', 'delimiter', '\n');

	X = load(fn_mfcc);
	X = X';
	X = X(1:D, :);

	for p=1:length(phns)
		start = max(starts(p)/128+1,1);
		end1 = min((ends(p)/128) + 1, length(X));
		phn = char(phns(p));
		if strcmp(phn, 'h#')
			phn = 'sil';
		end

		hmm_fields = fieldnames(HMM);

		max_p = {};
		max_p.phn = '';
		max_p.val = -Inf;

		for h=1:length(hmm_fields)
			hmm_p = char(hmm_fields{h});

			val = loglikHMM(HMM.(hmm_p), X(:, start:end1));

			if val > max_p.val
				max_p.phn = hmm_p;
				max_p.val = val;
			end

		end

		if strcmp(max_p.phn, phn)
			res = ['Correct Result: ', phn];
			correct = correct + 1;
		else
			res = ['Wrong. Expected: ', phn, ' computed: ', max_p.phn];
			wrong = wrong + 1;
		end

		total = total + 1;

		fprintf(fID, '%s\n', res);

	end


end

acc = (correct*100)/total

accuracy = ['accuracy is: ', int2str(correct), '/', int2str(total), ' = ', int2str(acc)];
fprintf(fID, '%s\n', accuracy);


fclose(fID);
