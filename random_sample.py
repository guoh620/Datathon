# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 21:24:54 2026

@author: 郭浩
"""

import pandas as pd
import numpy as np

filename = r"D:\OneDrive - KU Leuven\Datathon\Data_set_A\Data Set A_ Spaced Repetition\learning_traces.13m.csv"
n_samples = 10000

reservoir = None
count = 0

for chunk in pd.read_csv(filename, chunksize=50000):

    m = len(chunk)
    idx = np.arange(count, count + m)

    # probability that each of the m new rows should be included
    probs = n_samples / (idx + 1)

    mask = np.random.rand(m) < probs
    selected = chunk[mask]

    if selected.empty:
        count += m
        continue

    if reservoir is None:
        reservoir = selected
    else:
        # combine reservoir and new-selected rows, then keep only n_samples
        reservoir = (
            pd.concat([reservoir, selected])
            .sample(n=n_samples, replace=False)
            .reset_index(drop=True)
        )

    count += m

reservoir.to_csv(r"D:\OneDrive - KU Leuven\Datathon\Data_set_A\Data Set A_ Spaced Repetition\sampled_10000.csv", index=False)