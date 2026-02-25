KU-Leuven Datathon 2026, the gc project
----
This repository contains the files needed to recreate our Bayesian Beta-Binomial model: an alternative to the Half Life Regression model from duolingo (https://github.com/duolingo/halflife-regression.git) for spaced repetition data, description of their dataset is also included in the README_SpacedRepetitionData.txt file. 

In addition, more information has been derived from the starting dataset:
- diff_language: hours needed to learn the language according to FSI studies
- POS: this is the percentage of people that correctly identified different parts of speech according to [1, Datathon.pdf]
- len: length of the word that is being learnt, for verbs the infinitive form has been used.
These variables can be obtained by running the MAP.py code on the original dataset "learning_traces.13m.csv".

The model is used to derive the probability of a user remembering a word in a certain language, given that said word belongs to a specific POS category and has a certain length. Additional information on the derivation of the model can be found in the Datathon_v8_test_results.pdf.

If only a part of the data wants to be used instead of the full dataset, the random_sample.py can be used to create a subset. 
