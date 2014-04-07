SpeechRecognition
=================

Speaker and speech recognition. There is a dependency on FullBNT which is not included.

Speaker Identification (GMM)
----------------------

Speaker identification is the task of correctly identifying speaker sc from among S possible speakers si=1..S given an input speech sequence X, consisting of a succession of d-dimensional real vectors. d used here is 14. This is a discrete classification task (choosing among several speakers) that uses continuous-valued data (the vectors of real numbers) as input.

This uses M-component Gaussian mixture model (GMM) for each of the speakers in the Training data set.

gmmClassify.m calculates and reports the likelihoods of the five most likely speakers for each test utterance. This output is the folder unkn in individual files.


Speech Recogonition (HMM)
-------------------
Speech recognition is the task of correctly identifying a word sequence given an input speech sequence X. Typically this process involves language models, dictionaries, and grammars. This considers only a small subset of the acoustic modelling component and uses the Bayes Net Toolbox.

Word Error Rates using Levenshtein distance is done in Levenshtein.m


Additionally, there is code for PCA to reduce the dimensions of the data in pca.m