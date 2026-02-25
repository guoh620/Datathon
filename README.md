KU-Leuven Datathon 2026, the gc project
----
This repository contains the files needed to recreate our model: an alternative to the Half Life Regression model from duolingo (https://github.com/duolingo/halflife-regression.git) for spaced repetition data, description of their data is also included in the README_SpacedRepetitionData.txt file. 

In addition to their data, additional data has been computed from the starting dataset:
- diff_language: hours needed to learn the language according to FSI studies
- POS: this is the percentage of people that correctly identified different parts of speech according to [1, Datathon.pdf]
- len: length of the word that is being learnt, for verbs the infinitive form has been used. 

The model is used to derive the probability of a user remembering a word in a certain language, given that said word belongs to a specific POS category and has a certain length.
