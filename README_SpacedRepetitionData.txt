Information about the dataset: 

https://github.com/duolingo/halflife-regression

The Duolingo Spaced Repetition Dataset was released to support research on computational modeling of human memory and personalized review scheduling. It was used to develop Duolingo’s Half-Life Regression (HLR) model, a parametric model designed to estimate the probability that a learner will correctly recall a word as a function of time.

The dataset contains approximately 13 million user–lexeme interaction records collected from learners studying multiple target languages with diverse native language backgrounds.

The fundamental observation unit is a (user, word) practice event, meaning that each row represents a single instance where a learner was tested on a specific lexical item.

Overview dataset
Each row in the dataset represents a single recall event for one
user–lexeme pair at a specific time. The dataset is designed to model
probability of recall as a function of time and prior exposure history.

Variable Descriptions

1.  p_recall Type: Float in [0,1] Meaning: Empirical recall probability
    at that practice event. Typically computed as: p_recall =
    session_correct / session_seen This allows probabilistic regression
    instead of binary classification.

2.  timestamp Type: Unix time (seconds) Meaning: Absolute time of the
    practice interaction.

3.  delta Type: Integer (seconds) Meaning: Time elapsed since the
    previous exposure to the same lexeme by the same user. This is the
    key variable for modeling forgetting curves.

4.  user_id Anonymized learner identifier. Used for personalization and
    learner-specific modeling.

5.  learning_language Language being learned (L2).

6.  ui_language Interface language (usually learner’s native language).
    Enables cross-linguistic modeling.

7.  lexeme_id Unique identifier for the lexical item.

8.  lexeme_string Morphologically annotated word representation.
    Example: lernt/lernen

Breakdown: lernt -> surface form lernen -> lemma -> lexical verb ->
present tense -> third person -> singular

Allows morphological feature extraction and grammatical difficulty
modeling.

9.  history_seen Total number of times the user has seen this lexeme
    before this event.

10. history_correct Number of times previously answered correctly.
    Historical accuracy can be computed as: history_correct /
    history_seen

11. session_seen Number of times seen within the current session.

12. session_correct Number correct within the current session. Used to
    compute p_recall.

Mathematical Model (Half-Life Regression)

Recall probability is modeled as:

    P(recall) = 2^(-t / h)

where: t = delta (time since last practice) h = half-life parameter

Half-life is modeled as:

    h = exp(w^T x)

where: x = feature vector (history, lexical, language features) w =
learned parameters

This connects the dataset to exponential decay models, survival
analysis, and cognitive memory modeling.

Nevertheless, these data do not limit data scientists in practics. A lot of interesting things can be done with it! Think out of the box, and impress us at the datathon pitches!
