function [SE IE DE LEV_DIST] =Levenshtein(hypothesis,annotation_dir)
% Input:
%	hypothesis: The path to file containing the the recognition hypotheses
%	annotation_dir: The path to directory containing the annotations
%			(Ex. the Testing dir containing all the *.txt files)
% Outputs:
%	SE: proportion of substitution errors over all the hypotheses
%	IE: proportion of insertion errors over all the hypotheses
%	DE: proportion of deletion errors over all the hypotheses
%	LEV_DIST: proportion of overall error in all hypotheses

[Starts, Ends, Sents] = textread(hypothesis, '%d %d %s', 'delimiter', '\n');

annot_prefix = 'unkn_';

SE = 0;
IE = 0;
DE = 0;
LEV_DIST = 0;
num_words = 0;

UP = 0;
LEFT = 1;
UPLEFT = 2;

for s = 1:length(Sents)
	fnAnnot = [annotation_dir, filesep, annot_prefix, int2str(s), '.txt'];
	[SA, EA, SenA] = textread(fnAnnot, '%d %d %s', 'delimiter', '\n');

	annotSent = char(SenA(1));
	hypSent = char(Sents(s));

	annotSent = strread(annotSent,'%s','delimiter', ' ');
	hypSent = strread(hypSent, '%s', 'delimiter', ' ');


	n = length(annotSent);
	m = length(hypSent);

	r = zeros(n+1, m+1);
	r(:,:) = Inf;
	r(1, 1) = 0;

	b = zeros(n+1, m+1);

	for i=2:n+1
		for j=2:m+1
			del = r(i-1, j)+1;
			sub = r(i-1, j-1) + ~strcmp(annotSent(i-1), hypSent(j-1));
			ins = r(i, j-1) + 1;

			r(i, j) = min(del, min(sub, ins));

			if r(i, j) == del
				b(i, j) = UP;
			elseif r(i,j) == ins
				b(i,j) = LEFT;
			else
				b(i,j) = UPLEFT;
			end
		end
	end

	% Do the backtraking:

	i = n+1;
	j = m+1;
	subs_e = 0;
	ins_e = 0;
	del_e = 0;
	while i>1 & j>1
		if b(i,j) == UP
			del_e = del_e + 1;
			i = i -1;
		elseif b(i,j) == LEFT
			ins_e = ins_e + 1;
			j = j - 1;
		else
			if ~strcmp(annotSent(i-1), hypSent(j-1))
				subs_e = subs_e + 1;
			end
			i = i-1;
			j = j-1;
		end
	end

	SE = SE + subs_e;
	DE = DE + del_e;
	IE = IE + ins_e;
	num_words = num_words + length(annotSent);
end

SE = SE/num_words;
DE = DE/num_words;
IE = IE/num_words;
LEV_DIST = SE + DE + IE;

end
