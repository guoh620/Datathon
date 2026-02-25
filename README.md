KU-Leuven Datathon 2026, the gc project
----
This repository contains the files needed to recreate our Bayesian Beta-Binomial model: an alternative to the Half Life Regression model from Duolingo (https://github.com/duolingo/halflife-regression.git) for spaced repetition data that can be found at the link. A description of their dataset is also included in the README_SpacedRepetitionData.txt file. 

In addition, more information has been derived from the starting dataset:
- diff_language: hours needed to learn the language according to FSI studies
- POS: this is the percentage of people who correctly identified different parts of speech according to [1, Datathon.pdf]
- len: length of the word that is being learnt, for verbs, the infinitive form has been used.
These variables can be obtained by running the MAP.py code on the original dataset "learning_traces.13m.csv".

The model is used to estimate the probability that a user will remember a word in a language they're learning after a given time, given that the word belongs to a specific part-of-speech (POS) category and has a specific length. Additional information on the model's derivation can be found in Datathon_v9.pdf.

Predictions of BBB have been computed for a test set of the data, with the latter part of the model_stream_fixed.py, and compared to the HLR with the roc_comparison.R code.

If only a part of the data wants to be used instead of the full dataset, the random_sample.py can be used to create a subset. 
